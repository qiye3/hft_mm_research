"""现有预测信号的校准、成交诊断、策略开发和冻结验证入口。"""

import argparse
from pathlib import Path


# 【非核心】阶段与输入明确，计算分别归属预测研究和原生执行研究模块。
def main() -> None:
    """队列独立执行每个信号、前向折和日期，失败由既有后台监督器恢复。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['inventory', 'oof', 'calibrate', 'probe', 'baseline', 'pilot', 'develop',
                                        'select', 'validate', 'stress', 'report'])
    parser.add_argument('--signal')
    parser.add_argument('--fold', choices=['oof_0', 'oof_1', 'oof_2'])
    parser.add_argument('--date')
    arguments = parser.parse_args()
    if arguments.stage in ['inventory', 'oof', 'calibrate']:
        from mm_research.strategies.forecast_research import (
            calibrate_and_export, inventory_replay_days, train_forward_signal,
        )
        if arguments.stage == 'inventory':
            inventory_replay_days(arguments.config)
        elif arguments.stage == 'oof':
            if arguments.signal is None or arguments.fold is None:
                parser.error('oof阶段需要--signal和--fold')
            train_forward_signal(arguments.config, arguments.signal, arguments.fold)
        else:
            calibrate_and_export(arguments.config)
    else:
        from mm_research.strategies.forecast_execution import run_execution_stage
        run_execution_stage(arguments.config, arguments.stage, arguments.date)


if __name__ == '__main__':
    main()
