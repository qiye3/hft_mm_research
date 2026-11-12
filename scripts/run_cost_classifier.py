"""费用后交易价值分类实验入口。"""

import argparse
from pathlib import Path

from mm_research.models.cost_classifier import export_cost_signals, train_cost_classifier


# 【非核心】训练和信号导出是独立可恢复任务，GPU由已有共享锁协调。
def main() -> None:
    """运行指定周期和方向的获利分类，随后导出明确为净收益bps的信号。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['train', 'export'])
    parser.add_argument('--horizon', type=int, choices=[180, 1800, 7200])
    parser.add_argument('--side', type=int, choices=[-1, 1])
    parser.add_argument('--seed', type=int, default=17)
    arguments = parser.parse_args()
    if arguments.stage == 'train':
        train_cost_classifier(arguments.config, arguments.horizon, arguments.side, arguments.seed)
    else:
        export_cost_signals(arguments.config)


if __name__ == '__main__':
    main()
