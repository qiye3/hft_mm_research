"""冻结获利分类器的原生连续执行验证入口。"""

import argparse
from pathlib import Path

from mm_research.strategies.cost_native import (
    cleanup_credential, convert_native_day, plan_native, prefetch_native, prepare_native_block, replay_native_block, report_native,
)


# 【非核心】每个区间、转换日和报告均可由已有队列单独恢复。
def main() -> None:
    """执行原生区间规划、缺失日期准备、连续回放与完整费用后账本核算。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['plan', 'data', 'convert', 'replay', 'cleanup', 'report', 'build', 'prefetch'])
    parser.add_argument('--block')
    parser.add_argument('--date')
    parser.add_argument('--exclude-blocks', nargs='*', default=[])
    arguments = parser.parse_args()
    if arguments.stage == 'plan':
        plan_native(arguments.config)
    elif arguments.stage == 'prefetch':
        prefetch_native(arguments.config, arguments.exclude_blocks)
    elif arguments.stage == 'data':
        prepare_native_block(arguments.config, arguments.block)
    elif arguments.stage == 'convert':
        convert_native_day(arguments.config, arguments.date)
    elif arguments.stage == 'replay':
        replay_native_block(arguments.config, arguments.block)
    elif arguments.stage == 'cleanup':
        cleanup_credential(arguments.config)
    elif arguments.stage == 'build':
        from mm_research.strategies.defensive_quotes import build_defensive

        build_defensive(arguments.config)
    else:
        report_native(arguments.config)


if __name__ == '__main__':
    main()
