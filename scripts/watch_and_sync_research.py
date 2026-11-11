"""本地轻量同步器，服务器后台研究结束后取回汇总及结果文档。"""

import argparse
import datetime as dt
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
import fcntl
import json
import os
from pathlib import Path
import shlex
import smtplib
import ssl
import subprocess
import tempfile
import time


# 【非核心】读取服务器后台状态，连接错误交由外层有限等待处理。
def background_state(host: str, filename: Path) -> dict:
    """仅传输状态JSON，不读取原始行情或完整预测矩阵。"""
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
         shlex.join(['cat', str(filename)])],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, check=True, timeout=60,
    )
    return json.loads(result.stdout)


# 【非核心】先复制到临时目录，再逐文件原子发布，完成标记最后写入。
def sync_folder(host: str, remote_folder: Path, local_root: Path) -> Path:
    """只同步结果汇总或队列诊断，完整因子与模型数据继续保存在服务器。"""
    local_root.mkdir(parents=True, exist_ok=True)
    destination = local_root / remote_folder.name
    with tempfile.TemporaryDirectory(prefix='.sync_', dir=local_root) as temporary:
        subprocess.run(
            ['scp', '-r', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
             f'{host}:{remote_folder}', temporary],
            stdin=subprocess.DEVNULL, check=True, timeout=600,
        )
        received = Path(temporary) / remote_folder.name
        files = sorted(received.rglob('*'), key=lambda path: (path.name == 'completed.json', str(path)))
        for source in files:
            if source.is_file():
                output = destination / source.relative_to(received)
                output.parent.mkdir(parents=True, exist_ok=True)
                source.replace(output)
    return destination


# 【非核心】成功报告同步后发送QQ邮箱通知，同一结果目录通过锁和回执避免重复发送。
def send_completion_email(
    local_result: Path, docs_report: Path, password_file: Path, sender: str, recipient: str,
) -> None:
    """读取本机授权码，通过加密SMTP发送完成通知，回执只记录发送时间和邮件标识。"""
    receipt = local_result / 'mail_sent.json'
    with (local_result / 'mail.lock').open('a') as descriptor:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if receipt.exists():
            print('该研究的完成邮件已发送', flush=True)
            return
        message = EmailMessage()
        message['From'] = sender
        message['To'] = recipient
        message['Subject'] = 'HFT研究任务已全部成功完成'
        message['Date'] = formatdate(localtime=True)
        message['Message-ID'] = make_msgid()
        message.set_content(
            f"研究批次 {local_result.name} 已全部成功完成。\n\n"
            f"结果报告已同步到你的电脑\n{docs_report}\n\n"
            f"{docs_report.read_text()[:3000]}\n"
        )
        password = password_file.read_text().strip()
        connection = smtplib.SMTP_SSL('smtp.qq.com', 465, context=ssl.create_default_context(), timeout=30)
        try:
            connection.login(sender, password)
            refused = connection.send_message(message)
            if refused:
                raise smtplib.SMTPRecipientsRefused(refused)
            temporary = receipt.with_suffix('.partial.json')
            temporary.write_text(json.dumps({
                'status': 'accepted_by_smtp',
                'sent_at': dt.datetime.now().astimezone().isoformat(),
                'message_id': str(message['Message-ID']),
                'recipient': recipient,
            }, ensure_ascii=False, indent=2) + '\n')
            temporary.replace(receipt)
        finally:
            connection.close()
        print('完成邮件已由QQ邮件服务器接收', recipient, flush=True)


# 【非核心】等待服务器终态，自动同步到本地并写入用户指定结果文档。
def main() -> None:
    """连接中断每五分钟重试，等待上限24小时，计算任务始终由服务器执行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host')
    parser.add_argument('remote_result', type=Path)
    parser.add_argument('local_result', type=Path)
    parser.add_argument('docs_report', type=Path)
    parser.add_argument('--poll-seconds', type=int, default=300)
    parser.add_argument('--max-hours', type=float, default=24)
    parser.add_argument('--smtp-password-file', type=Path, help='仅含QQ邮箱SMTP授权码的本机私有文件')
    parser.add_argument('--mail-from', help='QQ邮箱发件地址')
    parser.add_argument('--mail-to', help='成功完成通知的收件地址')
    arguments = parser.parse_args()
    mail_options = [arguments.smtp_password_file, arguments.mail_from, arguments.mail_to]
    if any(mail_options) and not all(mail_options):
        parser.error('邮件通知需要同时指定 --smtp-password-file、--mail-from 和 --mail-to')
    arguments.local_result.mkdir(parents=True, exist_ok=True)
    (arguments.local_result / 'result_sync.pid').write_text(str(os.getpid()) + '\n')
    deadline = time.time() + arguments.max_hours * 3600
    while time.time() < deadline:
        try:
            state = background_state(arguments.host, arguments.remote_result / 'queue/background.json')
            if state['status'] == 'running':
                print(dt.datetime.now().astimezone().isoformat(), '服务器仍在计算', flush=True)
                time.sleep(arguments.poll_seconds)
                continue
            if state['status'] not in ['success', 'completed_with_failures', 'controller_failed']:
                raise ValueError(f"未知后台状态 {state['status']}")
            success = state['status'] == 'success'
            name = 'summary' if success else 'queue'
            result = sync_folder(arguments.host, arguments.remote_result / name, arguments.local_result)
            if success:
                report = (result / 'report.md').read_text()
                for figure in result.glob('*.png'):
                    report = report.replace(f']({figure.name})', f']({figure.resolve()})')
            else:
                report = '# 后台研究尚有失败任务\n\n' + (result / 'progress.md').read_text()
            arguments.docs_report.parent.mkdir(parents=True, exist_ok=True)
            temporary = arguments.docs_report.with_suffix('.partial.md')
            temporary.write_text(report)
            temporary.replace(arguments.docs_report)
            print('研究结果已同步', arguments.docs_report, flush=True)
            if not success:
                raise SystemExit(1)
            if arguments.smtp_password_file:
                send_completion_email(
                    arguments.local_result, arguments.docs_report,
                    arguments.smtp_password_file, arguments.mail_from, arguments.mail_to,
                )
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            print('连接或同步中断，下次继续', type(error).__name__, flush=True)
            time.sleep(arguments.poll_seconds)
    raise SystemExit('等待超过24小时，请检查服务器后台状态和同步日志')


if __name__ == '__main__':
    main()
