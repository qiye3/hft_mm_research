"""服务器后台入口，监督进程异常退出时恢复同一队列，正常终态自动结束。"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from mm_research.data.storage import write_json_atomic


# 【非核心】监督进程最多启动三次，任务重试预算由主队列独立管理。
def main() -> None:
    """运行队列至真实终态，保留后台进程PID、控制器退出码与最终状态。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(arguments.config.read_text())
    folder = root / config['output']
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'background.lock').open('a') as descriptor:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (folder / 'background.pid').write_text(str(os.getpid()) + '\n')
        for attempt in range(1, 4):
            write_json_atomic(folder / 'background.json', {
                'status': 'running', 'pid': os.getpid(), 'controller_attempt': attempt,
                'updated_at': time.time(),
            })
            completed = subprocess.run(
                [sys.executable, '-u', 'scripts/run_overnight.py', str(arguments.config)], cwd=root,
            )
            status_file = folder / 'status.json'
            if status_file.exists():
                state = json.loads(status_file.read_text())
                terminal = {'success', 'failed', 'skipped', 'memory_stopped'}
                if all(record['status'] in terminal for record in state['tasks'].values()):
                    success = all(record['status'] == 'success' for record in state['tasks'].values())
                    write_json_atomic(folder / 'background.json', {
                        'status': 'success' if success else 'completed_with_failures',
                        'pid': os.getpid(), 'controller_attempt': attempt,
                        'controller_return_code': completed.returncode, 'updated_at': time.time(),
                    })
                    if not success:
                        raise SystemExit(1)
                    return
            print('监督进程异常结束', attempt, completed.returncode, flush=True)
            if attempt < 3:
                time.sleep(30)
        write_json_atomic(folder / 'background.json', {
            'status': 'controller_failed', 'pid': os.getpid(), 'controller_attempt': 3,
            'controller_return_code': completed.returncode, 'updated_at': time.time(),
        })
        raise SystemExit(1)


if __name__ == '__main__':
    main()
