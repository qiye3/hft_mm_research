"""跨市场环境策略稳定性实验入口。"""

import argparse
from pathlib import Path

from mm_research.strategies.stable_strategy import screen_stability, select_stable, review_stable


# 【非核心】开发、冻结和复核分开执行，复核无法绕过冻结选择。
def main() -> None:
    """配置声明全部输入和候选，入口只调度指定阶段。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['screen', 'select', 'review'])
    parser.add_argument('--group')
    arguments = parser.parse_args()
    if arguments.stage == 'screen':
        if arguments.group is None:
            parser.error('screen需要--group')
        screen_stability(arguments.config, arguments.group, 'val_dev')
    elif arguments.stage == 'select':
        select_stable(arguments.config)
    else:
        review_stable(arguments.config)


if __name__ == '__main__':
    main()
