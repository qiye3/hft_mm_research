"""GPU树实验入口，通过公共GPU锁与序列训练互斥。"""
import argparse
import fcntl
from pathlib import Path

from mm_research.models.tree180 import run_tree


# 【非核心】只锁GPU计算阶段，CPU因子和序列缓存准备可继续运行。
def main() -> None:
    """读取模型配置和GPU锁位置，运行一个独立树模型实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--gpu-lock', type=Path, required=True)
    arguments = parser.parse_args()
    with arguments.gpu_lock.open('a') as descriptor:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        run_tree(arguments.config)


if __name__ == '__main__':
    main()
