"""180秒信号尾部、成本筛选和原生成交归因的后台任务入口。"""

import argparse
from pathlib import Path


# 【非核心】按明确阶段调用研究模块，队列独立记录各信号和时间分区的状态。
def main() -> None:
    """YAML声明输入版本与输出目录，单个任务失败由既有后台监督器恢复。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['thresholds', 'tail', 'tail_summary', 'extend_inputs', 'extend_signal',
                                        'screen', 'select', 'mechanism', 'report'])
    parser.add_argument('--signal')
    parser.add_argument('--split', choices=['val_dev', 'val_select', 'test'])
    parser.add_argument('--date')
    arguments = parser.parse_args()
    from mm_research.strategies.tradeability import (
        analyze_signal_tails, freeze_tail_thresholds, summarize_tails,
    )
    if arguments.stage == 'thresholds':
        freeze_tail_thresholds(arguments.config)
    elif arguments.stage == 'tail':
        if arguments.signal is None or arguments.split is None:
            parser.error('tail需要--signal和--split')
        analyze_signal_tails(arguments.config, arguments.signal, arguments.split)
    elif arguments.stage == 'tail_summary':
        summarize_tails(arguments.config)
    elif arguments.stage in ['extend_inputs', 'extend_signal']:
        from mm_research.strategies.tradeability_signals import export_historical_signal, prepare_historical_inputs
        if arguments.stage == 'extend_inputs':
            prepare_historical_inputs(arguments.config)
        else:
            if arguments.signal is None:
                parser.error('extend_signal需要--signal')
            export_historical_signal(arguments.config, arguments.signal)
    else:
        from mm_research.strategies.tradeability_execution import run_tradeability_stage
        run_tradeability_stage(arguments.config, arguments.stage, arguments.signal, arguments.split, arguments.date)


if __name__ == '__main__':
    main()
