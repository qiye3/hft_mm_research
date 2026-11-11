"""带截止时间、内存准入和阶段依赖的整夜实验队列。"""

import datetime as dt
import json
from pathlib import Path
import time

from mm_research.data.storage import write_json_atomic
from mm_research.overnight import memory_state, start_task, task_processes, task_anonymous_bytes, stop_task
from mm_research.multihorizon.common import context


# 【非核心】从明确的阶段、模型和折构造任务，线程和内存预算写入任务清单。
def research_tasks(root: Path, output: Path, source: Path, filename: Path) -> list[dict]:
    """CPU树、GPU网络与分析分别限流，先交付M1/M2研究，再扩展M3/M4和原始基线。"""
    tasks = []
    entry = str(source / 'scripts/run_multihorizon.py')
    python = str(root / '.venv-models/bin/python')
    for fold in ['oof_0', 'oof_1', 'oof_2', 'final']:
        identifier = f'prepare_{fold}'
        tasks.append({'id': identifier, 'lane': 'cpu', 'stage': 'train', 'model': 'preprocess', 'fold': fold,
                      'requires_success': ['e0'], 'threads': 4, 'estimated_peak_gib': 8, 'memory_limit_gib': 16,
                      'priority': 0})
        for model in ['M1', 'M2', 'M3', 'M4', 'M0']:
            dependencies = [identifier]
            if model in ['M3', 'M4']:
                dependencies += ['M1_weights_checked', 'M2_oof_0']
            if model == 'M0':
                dependencies += [f'{primary}_{other_fold}' for primary in ['M1', 'M2']
                                 for other_fold in ['oof_0', 'oof_1', 'oof_2', 'final']]
            tasks.append({'id': f'{model}_{fold}', 'lane': 'cpu' if model == 'M1' else 'gpu',
                          'stage': 'train', 'model': model, 'fold': fold, 'requires_success': dependencies,
                          'threads': 6 if model == 'M1' else 2, 'estimated_peak_gib': 18 if model == 'M1' else 10,
                          'memory_limit_gib': 26 if model == 'M1' else 18,
                          'priority': {'M1': 10, 'M2': 10, 'M3': 30, 'M4': 40, 'M0': 50}[model]})
    primary = [f'{model}_{fold}' for model in ['M1', 'M2'] for fold in ['oof_0', 'oof_1', 'oof_2', 'final']]
    extended = [f'{model}_{fold}' for model in ['M0', 'M3', 'M4'] for fold in ['oof_0', 'oof_1', 'oof_2', 'final']]
    for prefix, dependencies, priority in [('primary', primary, 20), ('extended', primary + extended, 60)]:
        for stage, previous in [('evaluate', None), ('persistence', 'evaluate'), ('inventory', 'persistence'), ('execution', 'inventory')]:
            if stage == 'inventory':
                previous = 'evaluate'
            required = dependencies if previous is None else [f'{prefix}_{previous}']
            tasks.append({'id': f'{prefix}_{stage}', 'lane': 'execution' if stage == 'execution' else 'analysis', 'stage': stage,
                          'requires_success': required, 'threads': 2, 'estimated_peak_gib': 10,
                          'memory_limit_gib': 22, 'priority': priority})
    tasks.append({'id': 'early_opportunities', 'lane': 'opportunities', 'stage': 'opportunities',
                  'requires_success': ['primary_evaluate'], 'threads': 2, 'estimated_peak_gib': 4,
                  'memory_limit_gib': 10, 'priority': 21})
    tasks.append({'id': 'early_baselines', 'lane': 'baselines', 'stage': 'baselines',
                  'requires_success': ['primary_evaluate'], 'threads': 2, 'estimated_peak_gib': 3,
                  'memory_limit_gib': 8, 'priority': 21})
    for task in tasks:
        task['maximum_seconds'] = {'evaluate': 1800, 'persistence': 3600, 'inventory': 7200,
                                   'train': 5400, 'opportunities': 25200, 'baselines': 25200,
                                   'execution': 32400}[task['stage']]
        command = [python, '-u', entry, str(filename), task['stage']]
        if task['stage'] == 'train':
            command += ['--model', task['model'], '--fold', task['fold']]
        task.update(command=command, after_finished=[])
    write_json_atomic(output / 'night/tasks.json', tasks)
    return tasks


# 【非核心】结束本轮专属下载任务并删除用户批准的临时凭证，原始网盘和历史结果独立保留。
def close_archive_session(output: Path) -> None:
    """截止或自然完成后清理授权会话，PID与本轮目录需匹配。"""
    runtime_file = output / 'l2/runtime.json'
    if runtime_file.exists():
        runtime = json.loads(runtime_file.read_text())
        command_file = Path('/proc') / str(runtime['pid']) / 'cmdline'
        if command_file.exists():
            command = command_file.read_bytes()
            if b'prepare_l2_archive.py' not in command or output.name.encode() not in command:
                raise ValueError('截止清理的L2进程归属不匹配')
            stop_task(runtime['pid'])
        credential = Path(runtime['credential_file'])
        credential.unlink(missing_ok=True)
        write_json_atomic(output / 'l2/credential_cleanup.json', {'temporary_credential_removed': True})


# 【核心逻辑266】按真实依赖运行独立实验，截止时保存已完成结果并终止本队列在跑任务。
def run_night(filename: Path) -> None:
    """只管理本次任务树，按匿名内存准入；失败不转换成零指标，也不影响独立候选。"""
    import fcntl
    root, config, output = context(filename)
    folder = output / 'night'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import os
    (folder / 'supervisor.pid').write_text(str(os.getpid()))
    source = Path(__file__).resolve().parents[3]
    tasks = research_tasks(root, output, source, filename.resolve())
    deadline = dt.datetime.fromisoformat(config['stop_at']).timestamp()
    state_file = folder / 'status.json'
    records = json.loads(state_file.read_text())['tasks'] if state_file.exists() else {'e0': {'status': 'waiting'}}
    for task in tasks:
        records.setdefault(task['id'], {'status': 'waiting'})
    records.setdefault('M1_weights_checked', {'status': 'waiting'})
    terminal = {'success', 'failed', 'skipped', 'memory_stopped', 'deadline_stopped', 'time_stopped'}
    with (folder / 'resources.jsonl').open('a') as resource_log:
        while True:
            now = time.time()
            if (output / 'e0/manifest.json').exists():
                records['e0']['status'] = 'success'
            if (output / 'e1/models/M1/oof_0/weights_complete.json').exists():
                records['M1_weights_checked']['status'] = 'success'
            elif records['M1_oof_0']['status'] in {'failed', 'memory_stopped'}:
                records['M1_weights_checked']['status'] = 'failed'
            memory = memory_state()
            for task in tasks:
                record = records[task['id']]
                if record['status'] != 'running':
                    continue
                pid = record['pid']
                anonymous = task_anonymous_bytes(pid)
                record['peak_anonymous_bytes'] = max(record['peak_anonymous_bytes'], anonymous)
                if not task_processes(pid):
                    exit_file = Path(record['exit_file'])
                    code = int(exit_file.read_text()) if exit_file.exists() else -1
                    record.update(status='success' if code == 0 else 'failed', exit_code=code, finished_at=now)
                    print('NIGHT_FINISHED', task['id'], record['status'], flush=True)
                elif (now >= deadline or anonymous > task['memory_limit_gib'] * 1024 ** 3
                      or now - record['started_at'] > task['maximum_seconds']):
                    stop_task(pid)
                    status = 'time_stopped'
                    if now >= deadline:
                        status = 'deadline_stopped'
                    elif anonymous > task['memory_limit_gib'] * 1024 ** 3:
                        status = 'memory_stopped'
                    record.update(status=status, finished_at=now)
            if now >= deadline:
                for record in records.values():
                    if record['status'] == 'waiting':
                        record.update(status='skipped', reason='截止时间已到')
            else:
                for task in sorted(tasks, key=lambda value: value['priority']):
                    record = records[task['id']]
                    if record['status'] != 'waiting':
                        continue
                    dependencies = [records[name]['status'] for name in task['requires_success']]
                    if any(status in terminal - {'success'} for status in dependencies):
                        record.update(status='skipped', reason='必要上游任务失败')
                        continue
                    if not all(status == 'success' for status in dependencies):
                        continue
                    running = [other for other in tasks if records[other['id']]['status'] == 'running']
                    if any(other['lane'] == task['lane'] for other in running):
                        continue
                    reserved = sum(max(0, other['estimated_peak_gib'] * 1024 ** 3
                                       - task_anonymous_bytes(records[other['id']]['pid'])) for other in running)
                    memory = memory_state()
                    if memory['available_bytes'] - reserved < (task['estimated_peak_gib'] + 6) * 1024 ** 3:
                        continue
                    records[task['id']] = start_task(root, folder, task)
                    print('NIGHT_STARTED', task['id'], flush=True)
            state = {'tasks': records, 'updated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                     'deadline': config['stop_at'], 'memory': memory}
            write_json_atomic(state_file, state)
            resource_log.write(json.dumps({'time': now, **memory}) + '\n')
            resource_log.flush()
            archive_done = (output / 'l2/batch_completed.json').exists()
            if now >= deadline or (archive_done and all(record['status'] in terminal for record in records.values())):
                close_archive_session(output)
                from mm_research.multihorizon.report import write_report
                write_report(filename)
                write_json_atomic(folder / 'completed.json', state)
                break
            time.sleep(10)
