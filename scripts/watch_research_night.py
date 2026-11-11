"""守护本轮研究与下载协调器，恢复异常退出，并在配置截止时间独立清理任务及临时凭证。"""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import context
from mm_research.multihorizon.night import close_archive_session
from mm_research.overnight import stop_task, task_processes


# 【非核心】用本轮路径和程序名核对进程归属，防止恢复或清理时误操作其他研究任务。
def owned_process(pid: int, output: Path, entry: str) -> bool:
    """PID需仍存活且命令行同时包含本轮目录及指定入口。"""
    command_file = Path('/proc') / str(pid) / 'cmdline'
    if not command_file.exists():
        return False
    command = command_file.read_bytes()
    return output.name.encode() in command and entry.encode() in command and bool(task_processes(pid))


# 【非核心】通过现有PID文件接管协调器，意外退出最多恢复三次，重复失败保存原因后停止重启。
def ensure_coordinator(root: Path, output: Path, filename: Path, credential: Path,
                       name: str, restarts: dict) -> None:
    """任务本身的失败由研究队列隔离，守护器只恢复负责排队的父进程。"""
    source = Path(__file__).resolve().parent
    if name == 'research':
        completed = output / 'night/completed.json'
        pid_file = output / 'night/supervisor.pid'
        entry = 'run_multihorizon.py'
        command = [sys.executable, '-u', str(source / entry), str(filename), 'night']
        log_file = output / 'night.log'
        pid = int(pid_file.read_text()) if pid_file.exists() else 0
    else:
        completed = output / 'l2/batch_completed.json'
        runtime = output / 'l2/runtime.json'
        entry = 'prepare_l2_archive.py'
        command = [sys.executable, '-u', str(source / entry), '--credential-file', str(credential),
                   '--output', str(output / 'l2'), '--batch']
        log_file = output / 'l2/batch.log'
        pid = json.loads(runtime.read_text())['pid'] if runtime.exists() else 0
    if completed.exists() or owned_process(pid, output, entry):
        return
    if restarts[name] >= 3:
        return
    if name == 'archive':
        # 协调器异常退出后，旧日期进程可能仍存活；先结束本轮孤立任务，避免重复写同一天。
        for process in Path('/proc').iterdir():
            if not process.name.isdigit():
                continue
            try:
                command_line = (process / 'cmdline').read_bytes()
            except FileNotFoundError:
                continue
            if b'--date\x00' in command_line and owned_process(int(process.name), output, entry):
                stop_task(int(process.name))
    environment = os.environ.copy()
    for variable in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
        environment.pop(variable, None)
    environment.update(OPENBLAS_NUM_THREADS='1', POLARS_MAX_THREADS='2', OMP_NUM_THREADS='1', NO_PROXY='*')
    with log_file.open('a') as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    restarts[name] += 1
    print('WATCHDOG_RESTART', name, process.pid, restarts[name], flush=True)


# 【核心逻辑289】独立守护下载及研究队列，协调器恢复沿用检查点，到08:00停止本轮任务并删除凭证。
def watch_night(filename: Path, credential: Path) -> None:
    """SSH或聊天退出不影响运行，单协调器连续失败三次后保留日志并继续守护其他协调器。"""
    root, config, output = context(filename)
    folder = output / 'watchdog'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'watchdog.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (folder / 'pid').write_text(str(os.getpid()))
    state_file = folder / 'status.json'
    restarts = json.loads(state_file.read_text())['restarts'] if state_file.exists() else {'research': 0, 'archive': 0}
    deadline = dt.datetime.fromisoformat(config['stop_at']).timestamp()
    while time.time() < deadline and not (output / 'night/completed.json').exists():
        for name in ['research', 'archive']:
            ensure_coordinator(root, output, filename, credential, name, restarts)
        release_completed_opportunity_days(output)
        write_json_atomic(state_file, {'restarts': restarts, 'updated_at': time.time(),
                                      'stop_at': config['stop_at'], 'status': 'watching'})
        time.sleep(30)
    pid_file = output / 'night/supervisor.pid'
    if pid_file.exists():
        pid = int(pid_file.read_text())
        if owned_process(pid, output, 'run_multihorizon.py'):
            stop_task(pid)
    status_file = output / 'night/status.json'
    if status_file.exists():
        state = json.loads(status_file.read_text())
        for record in state['tasks'].values():
            if record['status'] == 'running' and owned_process(record['pid'], output, 'run_multihorizon.py'):
                stop_task(record['pid'])
                record.update(status='deadline_stopped', finished_at=time.time())
            elif record['status'] == 'waiting':
                record.update(status='skipped', reason='截止时间已到')
        write_json_atomic(status_file, state)
    try:
        close_archive_session(output)
    finally:
        credential.unlink(missing_ok=True)
        lock.close()
    write_json_atomic(folder / 'completed.json', {'restarts': restarts, 'stopped_at': time.time(),
                                                'temporary_credential_removed': not credential.exists()})
    from mm_research.multihorizon.report import write_report
    write_report(filename)


# 【非核心】训练日的全部固定机会完成后即可释放事件缓存，后续价值估计只读取已保存账本。
def release_completed_opportunity_days(output: Path) -> None:
    """仅发出已消费标记，实际目录归属核查与删除仍由下载协调器执行。"""
    plan_file = output / 'e4/opportunity_case_plan.json'
    if not plan_file.exists():
        return
    by_date = {}
    for case in json.loads(plan_file.read_text()):
        by_date.setdefault(case['date'], []).append(case['id'])
    for date, identifiers in by_date.items():
        completed = all((output / 'e4/runs' / identifier / 'execution_metrics.json').exists()
                        for identifier in identifiers)
        if completed:
            write_json_atomic(output / 'l2/consumed' / f'{date}.json',
                              {'status': 'all_training_opportunity_cases_complete', 'completed_cases': len(identifiers),
                               'remaining_consumers': 'saved_ledgers_and_signals_only'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--credential-file', type=Path, required=True)
    arguments = parser.parse_args()
    watch_night(arguments.config.resolve(), arguments.credential_file)
