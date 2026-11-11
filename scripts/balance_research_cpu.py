"""并行研究任务的CPU分配，减少容器总配额耗尽造成的GPU供数等待。"""

import argparse
import json
import os
from pathlib import Path
import time

from mm_research.data.storage import write_json_atomic
from mm_research.overnight import task_processes


# 【非核心】给已启动任务及当前子进程固定CPU集合，后续子进程继承其父进程设置。
def assign_task_cpus(records: dict, cpus: list[int]) -> int:
    """运行任务数由队列限制，进程退出竞争仅跳过已消失的进程。"""
    running = [record for record in records.values() if record['status'] == 'running']
    for index, record in enumerate(running):
        start = index * len(cpus) // len(running)
        stop = (index + 1) * len(cpus) // len(running)
        assigned = set(cpus[start:stop])
        for pid in task_processes(record['pid']):
            try:
                for thread in (Path('/proc') / str(pid) / 'task').iterdir():
                    os.sched_setaffinity(int(thread.name), assigned)
            except (ProcessLookupError, FileNotFoundError):
                pass
    return len(running)


# 【非核心】CPU因子八个单核槽与GPU两个CPU槽共享十核配额，持续跟随队列的新任务。
def main() -> None:
    """两队列全部结束后退出，资源策略与更新时间单独保存。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('factor_status', type=Path)
    parser.add_argument('gpu_status', type=Path)
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    terminal = {'success', 'failed', 'skipped', 'memory_stopped'}
    while True:
        factors = json.loads(arguments.factor_status.read_text())
        models = json.loads(arguments.gpu_status.read_text())
        models_finished = all(record['status'] in terminal for record in models['tasks'].values())
        factor_cpus = allowed[:10] if models_finished else allowed[:8]
        model_cpus = [] if models_finished else allowed[8:10]
        factor_count = assign_task_cpus(factors['tasks'], factor_cpus)
        model_count = assign_task_cpus(models['tasks'], model_cpus)
        write_json_atomic(arguments.output, {
            'updated_at': time.time(), 'factor_cpus': factor_cpus,
            'gpu_feed_cpus': model_cpus,
            'running_factor_tasks': factor_count, 'running_gpu_tasks': model_count,
        })
        records = [*factors['tasks'].values(), *models['tasks'].values()]
        if all(record['status'] in terminal for record in records):
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
