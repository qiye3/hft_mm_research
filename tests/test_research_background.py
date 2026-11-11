"""后台重试预算、依赖恢复和最终状态报告。"""

import pytest

from mm_research.overnight import schedule_retries, write_queue_report


# 【非核心】失败重试先恢复依赖等待，冷却期间保持同一次重试。
def test_retry_recovers_dependency_and_preserves_history():
    """单任务最多两次重试，第三次失败保持真实失败终态。"""
    tasks = {
        'evaluate': {'max_retries': 2, 'retry_delay_seconds': 60, 'requires_success': []},
        'finalize': {'requires_success': ['evaluate']},
    }
    state = {'tasks': {
        'evaluate': {'status': 'failed', 'finished_at': 100, 'exit_code': 1, 'log': 'run.log'},
        'finalize': {'status': 'skipped', 'reason': '所需上游任务失败'},
    }}
    schedule_retries(state, tasks, 110)
    record = state['tasks']['evaluate']
    assert record['status'] == 'waiting'
    assert record['retry_at'] == 170
    assert record['retry_count'] == 1
    assert record['retry_history'][0]['exit_code'] == 1
    assert state['tasks']['finalize']['status'] == 'waiting'
    schedule_retries(state, tasks, 120)
    assert record['retry_count'] == 1
    record.update(status='failed', finished_at=200)
    schedule_retries(state, tasks, 210)
    assert record['retry_count'] == 2
    record.update(status='failed', finished_at=300)
    schedule_retries(state, tasks, 310)
    assert record['status'] == 'failed'
    assert record['retry_count'] == 2


# 【非核心】未开启重试的其他队列维持原行为。
def test_retry_disabled_by_default():
    """新增后台能力只对明确配置的任务生效。"""
    state = {'tasks': {'job': {'status': 'failed', 'finished_at': 1}}}
    schedule_retries(state, {'job': {'requires_success': []}}, 2)
    assert state['tasks']['job']['status'] == 'failed'


# 【非核心】失败终态必须保留在用户可读报告中。
def test_report_keeps_failed_terminal_status(tmp_path):
    """队列结束时保留失败名称与日志，不将终止误报为全部成功。"""
    state = {'updated_at': 1, 'tasks': {
        'good': {'status': 'success'},
        'bad': {'status': 'failed', 'log': '/server/bad.log', 'retry_count': 2},
    }}
    write_queue_report(tmp_path, state)
    report = (tmp_path / 'progress.md').read_text()
    assert '部分任务未成功' in report
    assert 'bad' in report
    assert '/server/bad.log' in report


# 【非核心】监督进程崩溃后从同一状态继续，工作终态结束恢复循环。
@pytest.mark.parametrize('terminal_status', ['success', 'failed', 'running'])
def test_background_recovers_controller_crash(tmp_path, monkeypatch, terminal_status):
    """第一次返回异常且任务运行中，第二次成功结束。"""
    import importlib.util
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch
    import yaml

    filename = Path(__file__).resolve().parents[1] / 'scripts/run_research_background.py'
    specification = importlib.util.spec_from_file_location('background_test', filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    queue = tmp_path / 'queue'
    queue.mkdir()
    config = tmp_path / 'research.yaml'
    config.write_text(yaml.safe_dump({'output': str(queue)}))
    monkeypatch.setattr(module.sys, 'argv', ['background', str(config)])
    attempts = []

    # 【非核心】模拟监督进程退出前发布的真实状态。
    def controller(*args, **kwargs):
        attempts.append(1)
        status = 'running' if len(attempts) == 1 else terminal_status
        (queue / 'status.json').write_text(json.dumps({'tasks': {'job': {'status': status}}}))
        return SimpleNamespace(returncode=1 if len(attempts) == 1 else 0)

    with patch.object(module.subprocess, 'run', side_effect=controller):
        with patch.object(module.time, 'sleep'):
            if terminal_status == 'success':
                module.main()
            else:
                with pytest.raises(SystemExit) as failure:
                    module.main()
                assert failure.value.code == 1
    expected_attempts = 3 if terminal_status == 'running' else 2
    assert len(attempts) == expected_attempts
    expected_status = {'success': 'success', 'failed': 'completed_with_failures', 'running': 'controller_failed'}
    assert json.loads((queue / 'background.json').read_text())['status'] == expected_status[terminal_status]


# 【非核心】后台完成后自动复制报告，图片链接指向实际本地文件。
def test_sync_publishes_report_and_local_links(tmp_path, monkeypatch):
    """模拟服务器终态与SCP传输，执行完整本地同步入口。"""
    import importlib.util
    from pathlib import Path
    from unittest.mock import patch

    filename = Path(__file__).resolve().parents[1] / 'scripts/watch_and_sync_research.py'
    specification = importlib.util.spec_from_file_location('sync_test', filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    local = tmp_path / 'results'
    report = tmp_path / 'docs/result.md'
    import sys
    monkeypatch.setattr(sys, 'argv', ['sync', 'host', '/remote/results', str(local), str(report)])

    # 【非核心】生成与最终汇总相同的目录边界，避免测试连接真实服务器。
    def transfer(command, **kwargs):
        assert command[0] == 'scp'
        received = Path(command[-1]) / 'summary'
        received.mkdir()
        (received / 'report.md').write_text('# 完成\n\n![图](chart.png)\n')
        (received / 'chart.png').write_bytes(b'image')
        (received / 'completed.json').write_text('{}')

    with patch.object(module, 'background_state', return_value={'status': 'success'}):
        with patch.object(module.subprocess, 'run', side_effect=transfer):
            module.main()
    assert (local / 'summary/completed.json').exists()
    assert str((local / 'summary/chart.png').resolve()) in report.read_text()


# 【非核心】邮件服务确认接收后写回执，重复调用直接复用回执。
def test_completion_mail_sends_once_without_saving_password(tmp_path):
    """SMTP走加密连接，凭证只从指定文件读取，回执不包含授权码。"""
    import importlib.util
    from pathlib import Path
    from unittest.mock import patch

    filename = Path(__file__).resolve().parents[1] / 'scripts/watch_and_sync_research.py'
    specification = importlib.util.spec_from_file_location('mail_test', filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    report = tmp_path / 'report.md'
    report.write_text('# 全部成功\n')
    password = tmp_path / 'smtp.txt'
    password.write_text('test-authorization-code')
    with patch.object(module.smtplib, 'SMTP_SSL') as smtp:
        smtp.return_value.send_message.return_value = {}
        module.send_completion_email(tmp_path, report, password, 'sender@qq.com', 'recipient@qq.com')
        module.send_completion_email(tmp_path, report, password, 'sender@qq.com', 'recipient@qq.com')
    smtp.assert_called_once()
    smtp.return_value.login.assert_called_once_with('sender@qq.com', 'test-authorization-code')
    assert smtp.return_value.send_message.call_count == 1
    assert 'test-authorization-code' not in (tmp_path / 'mail_sent.json').read_text()
