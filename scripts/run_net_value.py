"""直接费用后价值模型的GPU实验入口。"""

import argparse
from pathlib import Path

from mm_research.models.net_value import train_net_value


# 【非核心】显式接收目标周期、方向及训练损失，任务独立保存和恢复。
def main() -> None:
    """将冻结配置交给价值模型训练，不在入口修改数据或选择规则。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--horizon', type=int, choices=[1800, 7200], required=True)
    parser.add_argument('--side', type=int, choices=[-1, 1], required=True)
    parser.add_argument('--family', choices=['huber', 'clipped', 'mean'], required=True)
    arguments = parser.parse_args()
    train_net_value(arguments.config, arguments.horizon, arguments.side, arguments.family)


if __name__ == '__main__':
    main()
