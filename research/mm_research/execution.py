"""有界进程并发，避免任务结果在父进程内无限排队。"""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


# 【非核心】按输入顺序交付有限在途任务，保留结果顺序并限制内存。
def ordered_process_map(function, tasks, workers: int, *, recycle_workers: bool = False):
    """输入可序列化函数与任务，最多保留workers份在途结果；因子分块可用独立进程释放分配器。"""
    if workers < 1:
        raise ValueError("workers必须为正整数")
    if workers == 1 and not recycle_workers:
        yield from map(function, tasks)
        return
    tasks = iter(tasks)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1 if recycle_workers else None,
    ) as pool:
        pending = deque()
        for _ in range(workers):
            task = next(tasks, None)
            if task is None:
                break
            pending.append(pool.submit(function, task))
        while pending:
            yield pending.popleft().result()
            task = next(tasks, None)
            if task is not None:
                pending.append(pool.submit(function, task))
