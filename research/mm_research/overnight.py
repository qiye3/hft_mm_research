"""服务器常驻研究队列，限制匿名内存和并发，按任务隔离失败并保存恢复状态。"""

import fcntl
import datetime as dt
import json
import os
from pathlib import Path
import resource
import shlex
import signal
import subprocess
import time

import yaml

from mm_research.data.storage import write_json_atomic

GIB = 1024 ** 3


# 【非核心】读取容器实限，区分匿名内存和可回收文件缓存。
def memory_state() -> dict:
    """仅用于Linux服务器，脏页从可回收缓存中扣除，返回实际内存余量。"""
    cgroup = Path('/sys/fs/cgroup')
    limit = int((cgroup / 'memory.max').read_text())
    current = int((cgroup / 'memory.current').read_text())
    values = dict(line.split() for line in (cgroup / 'memory.stat').read_text().splitlines())
    values = {name: int(value) for name, value in values.items()}
    reclaimable = max(0, values['file'] - values['shmem'] - values['file_dirty'] - values['file_writeback'])
    return {'limit_bytes': limit, 'current_bytes': current, 'anonymous_bytes': values['anon'],
            'reclaimable_file_bytes': reclaimable, 'available_bytes': limit - current + reclaimable,
            'working_bytes': current - values['inactive_file']}


# 【非核心】取得一项任务的全部后代，正常退出中的进程不再计入。
def task_processes(pid: int) -> list[int]:
    """通过proc父子关系获取任务范围，不读取可能包含凭证的命令行。"""
    result = []
    pending = [pid]
    while pending:
        current = pending.pop()
        folder = Path('/proc') / str(current)
        try:
            state = (folder / 'stat').read_text().split(') ', 1)[1].split()[0]
            if state == 'Z':
                continue
            children = (folder / 'task' / str(current) / 'children').read_text().split()
        except FileNotFoundError:
            continue
        result.append(current)
        pending.extend(int(child) for child in children)
    return result


# 【非核心】累计进程树匿名RSS，避免把共享文件映射误认为重复堆内存。
def task_anonymous_bytes(pid: int) -> int:
    """读取各进程RssAnon，映射研究矩阵的可回收页由容器缓存统计单独监测。"""
    total = 0
    for process in task_processes(pid):
        try:
            lines = (Path('/proc') / str(process) / 'status').read_text().splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            if line.startswith('RssAnon:'):
                total += int(line.split()[1]) * 1024
                break
    return total


# 【非核心】终止一个明确任务进程树，给已注册退出处理留出时间。
def stop_task(pid: int) -> None:
    """只处理指定任务的后代，避免波及SSH、Jupyter和其他用户会话。"""
    processes = task_processes(pid)
    for process in processes:
        try:
            os.kill(process, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(2)
    for process in processes:
        try:
            os.kill(process, signal.SIGKILL)
        except ProcessLookupError:
            pass


# 【非核心】用硬匿名分配上限、独立会话和明确线程预算启动一项任务。
def start_task(root: Path, folder: Path, task: dict) -> dict:
    """子进程独立记录退出码，监督脚本重启后可重新接管，不依赖原SSH会话。"""
    if 'config_template' in task:
        generated = yaml.safe_load((root / task['config_template']).read_text())
        generated.update(task['config_overrides'])
        filename = root / task['generated_config']
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(yaml.safe_dump(generated, sort_keys=False))
    directory = folder / task['id']
    directory.mkdir(parents=True, exist_ok=True)
    exit_file = directory / 'exit_code.txt'
    exit_file.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONPATH='research', POLARS_MAX_THREADS=str(task['threads']),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMBA_NUM_THREADS=str(task['threads']), MALLOC_ARENA_MAX='2')
    for variable in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
        environment.pop(variable, None)
    environment.update(NO_PROXY='*', no_proxy='*')
    limit = int(task['memory_limit_gib'] * GIB)

    # 【非核心】RLIMIT_DATA约束堆与匿名映射，正式文件映射由全局内存监控管理。
    def allocation_limit():
        """在exec之前给任务及后代设置相同硬上限。"""
        resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))

    command = shlex.join(['/usr/bin/time', '-v', *task['command']])
    command += '\nresult=$?\nprintf "%s\\n" "$result" > ' + shlex.quote(str(exit_file)) + '\nexit "$result"'
    with (directory / 'run.log').open('a') as log:
        process = subprocess.Popen(['bash', '-c', command], cwd=root, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   preexec_fn=allocation_limit)
    return {'status': 'running', 'pid': process.pid, 'started_at': time.time(), 'exit_file': str(exit_file),
            'log': str(directory / 'run.log'), 'peak_anonymous_bytes': 0}


# 【非核心】读取主队列与独立扩展文件，任务ID必须保持唯一。
def load_queue_config(filename: Path) -> dict:
    """新增批次放入queue_directory，运行中只允许追加任务。"""
    config = yaml.safe_load(filename.read_text())
    root = Path(__file__).resolve().parents[2]
    for extension in sorted((root / config['queue_directory']).glob('*.yaml')):
        config['tasks'].extend(yaml.safe_load(extension.read_text())['tasks'])
    identifiers = [task['id'] for task in config['tasks']]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('夜间队列任务ID重复')
    return config


# 【核心逻辑24】常驻调度CPU与GPU队列，内存紧张时停止额外任务，独立任务失败后继续运行。
def run_overnight(config_path: Path) -> None:
    """配置明确任务依赖、并发、内存上限及模型时间预算，状态每十秒原子保存。"""
    root = Path(__file__).resolve().parents[2]
    config = load_queue_config(config_path)
    folder = root / config['output']
    folder.mkdir(parents=True, exist_ok=True)
    descriptor = (folder / 'supervisor.lock').open('a')
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (folder / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
    status_file = folder / 'status.json'
    if status_file.exists():
        state = json.loads(status_file.read_text())
        if state['config'] != config:
            raise ValueError('常驻队列配置变化，请使用新的output目录')
    else:
        records = {}
        for task in config['tasks']:
            if 'blocked_reason' in task:
                records[task['id']] = {'status': 'skipped', 'reason': task['blocked_reason']}
            elif 'adopt_pid' in task:
                process = Path('/proc') / str(task['adopt_pid'])
                try:
                    command = (process / 'cmdline').read_bytes().decode().replace('\0', ' ')
                except FileNotFoundError:
                    command = ''
                if task['adopt_command_marker'] not in command:
                    successful = (root / task['success_file']).exists()
                    records[task['id']] = {'status': 'success' if successful else 'failed',
                                            'reason': '原进程已结束，按明确交接产物记录状态'}
                    continue
                limit = int(task['memory_limit_gib'] * GIB)
                resource.prlimit(task['adopt_pid'], resource.RLIMIT_DATA, (limit, limit))
                records[task['id']] = {'status': 'running', 'pid': task['adopt_pid'], 'started_at': time.time(),
                                        'peak_anonymous_bytes': 0, 'adopted': True}
            else:
                records[task['id']] = {'status': 'waiting'}
        state = {'config': config, 'tasks': records, 'started_at': time.time()}
    tasks = {task['id']: task for task in config['tasks']}
    terminal = {'success', 'failed', 'skipped', 'memory_stopped'}
    previous_cpu = None
    previous_time = None
    with (folder / 'resources.jsonl').open('a') as resources:
        while True:
            updated = load_queue_config(config_path)
            if updated != config:
                if {key: value for key, value in updated.items() if key != 'tasks'} != {key: value for key, value in config.items() if key != 'tasks'}:
                    raise ValueError('运行中的监督参数发生变化，需要明确重启')
                incoming = {task['id']: task for task in updated['tasks']}
                if any(incoming.get(name) != task for name, task in tasks.items()):
                    raise ValueError('运行中只能追加任务，已有任务定义不能修改')
                for name, task in incoming.items():
                    if name not in tasks:
                        if 'blocked_reason' in task:
                            state['tasks'][name] = {'status': 'skipped', 'reason': task['blocked_reason']}
                        else:
                            state['tasks'][name] = {'status': 'waiting'}
                tasks = incoming
                config = updated
                state['config'] = config
                print('新增队列任务已载入', flush=True)
            memory = memory_state()
            cpu = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
            quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
            now = time.monotonic()
            used = int(cpu['usage_usec']) / 1e6
            if previous_cpu is not None:
                memory['cpu_utilization_pct'] = (used - previous_cpu) / (now - previous_time) / (int(quota) / int(period)) * 100
            previous_cpu, previous_time = used, now
            active = []
            for name, record in state['tasks'].items():
                task = tasks[name]
                if record['status'] != 'running':
                    continue
                pid = record['pid']
                if 'handoff_files' in task and all((root / filename).exists() for filename in task['handoff_files']):
                    stop_task(pid)
                    record.update(status='success', reason='交接产物已发布，后续任务按独立进程运行', finished_at=time.time())
                    continue
                anonymous = task_anonymous_bytes(pid)
                record['peak_anonymous_bytes'] = max(record['peak_anonymous_bytes'], anonymous)
                processes = task_processes(pid)
                if not processes:
                    if record.get('adopted', False):
                        successful = (root / task['success_file']).exists()
                        record['status'] = 'success' if successful else 'failed'
                    else:
                        exit_file = Path(record['exit_file'])
                        exit_code = int(exit_file.read_text()) if exit_file.exists() else -1
                        record.update(status='success' if exit_code == 0 else 'failed', exit_code=exit_code)
                    record['finished_at'] = time.time()
                    print(f"任务结束 {name} {record['status']}", flush=True)
                    continue
                active.append(name)
                if anonymous > task['memory_limit_gib'] * GIB:
                    stop_task(pid)
                    record.update(status='memory_stopped', reason='任务匿名内存超过上限', finished_at=time.time())
            # 留出真实可用余量；可回收干净缓存不触发停止，避免把正常缓存占用当成泄漏。
            if memory['available_bytes'] < config['emergency_reserve_gib'] * GIB:
                candidates = [name for name in active if not tasks[name].get('protected', False)
                              and state['tasks'][name]['status'] == 'running']
                if candidates:
                    victim = max(candidates, key=lambda name: task_anonymous_bytes(state['tasks'][name]['pid']))
                    stop_task(state['tasks'][victim]['pid'])
                    state['tasks'][victim].update(status='memory_stopped', reason='容器可用内存低于紧急预留', finished_at=time.time())
                    memory = memory_state()
            for name, task in sorted(tasks.items(), key=lambda item: item[1]['priority']):
                record = state['tasks'][name]
                if record['status'] != 'waiting':
                    continue
                required = task['requires_success']
                finished = task['after_finished']
                if any(state['tasks'][dependency]['status'] in terminal - {'success'} for dependency in required):
                    record.update(status='skipped', reason='所需上游任务失败', finished_at=time.time())
                    continue
                if not all(state['tasks'][dependency]['status'] == 'success' for dependency in required):
                    continue
                if not all(state['tasks'][dependency]['status'] in terminal for dependency in finished):
                    continue
                running_in_lane = sum(other['status'] == 'running' and tasks[other_name]['lane'] == task['lane']
                                      for other_name, other in state['tasks'].items())
                if running_in_lane >= config['lane_limits'][task['lane']]:
                    continue
                reserved = sum(max(tasks[active_name]['estimated_peak_gib'] * GIB
                                   - task_anonymous_bytes(active_record['pid']), 0)
                               for active_name, active_record in state['tasks'].items() if active_record['status'] == 'running')
                if memory['available_bytes'] - reserved < (task['estimated_peak_gib'] + config['start_reserve_gib']) * GIB:
                    continue
                state['tasks'][name] = start_task(root, folder, task)
                memory = memory_state()
                print(f'任务开始 {name}', flush=True)
            state['updated_at'] = time.time()
            state['memory'] = memory
            write_json_atomic(status_file, state)
            report_at = dt.datetime.fromisoformat(config['morning_report_at']).timestamp()
            if time.time() >= report_at and not state.get('morning_report_attempted', False):
                report = subprocess.run([str(root / '.venv/bin/python'), 'scripts/collect_night_results.py', str(config_path)], cwd=root)
                state['morning_report_attempted'] = True
                state['morning_report_written'] = report.returncode == 0
                state['morning_report_returncode'] = report.returncode
                write_json_atomic(status_file, state)
            resources.write(json.dumps({'time': time.time(), **memory}) + '\n')
            resources.flush()
            if all(record['status'] in terminal for record in state['tasks'].values()):
                write_json_atomic(folder / 'completed.json', state)
                report = subprocess.run([str(root / '.venv/bin/python'), 'scripts/collect_night_results.py', str(config_path)], cwd=root)
                state['final_report_returncode'] = report.returncode
                write_json_atomic(folder / 'completed.json', state)
                print('OVERNIGHT_QUEUE_COMPLETE', flush=True)
                break
            time.sleep(config['poll_seconds'])
