"""固定本金和敞口约束下的策略规模实验入口。"""

import argparse
from pathlib import Path

from mm_research.strategies.account_sizing import screen_account, select_and_review_account


# 【非核心】将规模开发和冻结复核分开调度，复核不得自行挑规模。
def main() -> None:
    """配置给出本金、敞口、规模网格及固定信号规则。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['screen', 'review'])
    parser.add_argument('--group')
    arguments = parser.parse_args()
    if arguments.stage == 'screen':
        if arguments.group is None:
            parser.error('screen需要--group')
        screen_account(arguments.config, arguments.group, 'val_dev')
    else:
        select_and_review_account(arguments.config)


if __name__ == '__main__':
    main()
