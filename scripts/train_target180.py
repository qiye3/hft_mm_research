"""180秒GPU模型的单实验入口，批次、失败隔离和恢复由研究队列负责。"""
import argparse
from pathlib import Path

from mm_research.models.target180 import run_target180


# 【非核心】显式传入配置、种子和学习率，避免隐式选择训练实验。
def main() -> None:
    """将参数交给研究包，所有输入输出由模型配置声明。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--learning-rate', type=float, required=True)
    arguments = parser.parse_args()
    run_target180(arguments.config, arguments.seed, arguments.learning_rate)


if __name__ == '__main__':
    main()
