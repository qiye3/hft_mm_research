"""较长周期、低换手交易研究入口，复用原高频输入和CUDA训练实现。"""

import argparse
from pathlib import Path

from mm_research.models.intraday import export_intraday, prepare_intraday, train_intraday


# 【非核心】入口只分发已声明的两个目标周期和三个种子。
def main() -> None:
    """执行过去状态构造、模型训练或组合信号导出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['prepare', 'train', 'export'])
    parser.add_argument('--horizon', type=int, choices=[1800, 7200])
    parser.add_argument('--seed', type=int, choices=[17, 42, 73])
    arguments = parser.parse_args()
    if arguments.stage == 'prepare':
        prepare_intraday(arguments.config)
    elif arguments.stage == 'train':
        train_intraday(arguments.config, arguments.horizon, arguments.seed)
    else:
        export_intraday(arguments.config)


if __name__ == '__main__':
    main()
