"""持续优化中的交易收益实验入口，计算与成交核算复用研究包。"""

import argparse
from pathlib import Path

from mm_research.strategies.profit_research import freeze_thresholds, report_profit, screen_group, select_rules


# 【非核心】独立阶段交给已有后台队列，模型、筛选和复核之间保持明确依赖。
def main() -> None:
    """运行训练门槛、开发筛选、冻结规则、历史复核和报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['thresholds', 'develop', 'select', 'review', 'report', 'wide_day', 'wide_select', 'wide_report',
                                        'defensive_build', 'defensive_prepare', 'defensive_day', 'defensive_select', 'defensive_report',
                                        'trend_develop', 'trend_select', 'trend_review', 'trend_report'])
    parser.add_argument('--group')
    parser.add_argument('--date')
    parser.add_argument('--split', choices=['val_dev', 'val_select'])
    parser.add_argument('--signal-column', default='prediction_log_bps')
    arguments = parser.parse_args()
    if arguments.stage == 'thresholds':
        freeze_thresholds(arguments.config)
    elif arguments.stage in ['develop', 'review']:
        split = 'val_dev' if arguments.stage == 'develop' else 'val_select'
        screen_group(arguments.config, arguments.group, split, arguments.signal_column)
    elif arguments.stage == 'select':
        select_rules(arguments.config)
    elif arguments.stage == 'report':
        report_profit(arguments.config)
    elif arguments.stage.startswith('wide_'):
        from mm_research.strategies.wide_quotes import report_wide, run_wide_day, select_wide

        if arguments.stage == 'wide_day':
            run_wide_day(arguments.config, arguments.date, arguments.split)
        elif arguments.stage == 'wide_select':
            select_wide(arguments.config)
        else:
            report_wide(arguments.config)
    elif arguments.stage.startswith('defensive_'):
        from mm_research.strategies.defensive_quotes import build_defensive, finish_defensive, prepare_defensive, run_defensive_day

        if arguments.stage == 'defensive_build':
            build_defensive(arguments.config)
        elif arguments.stage == 'defensive_prepare':
            prepare_defensive(arguments.config)
        elif arguments.stage == 'defensive_day':
            run_defensive_day(arguments.config, arguments.date, arguments.split)
        else:
            finish_defensive(arguments.config, arguments.stage)
    else:
        from mm_research.strategies.trend_research import report_trend, screen_trend, select_trend

        if arguments.stage in ['trend_develop', 'trend_review']:
            split = 'val_dev' if arguments.stage == 'trend_develop' else 'val_select'
            screen_trend(arguments.config, arguments.group, split)
        elif arguments.stage == 'trend_select':
            select_trend(arguments.config)
        else:
            report_trend(arguments.config)


if __name__ == '__main__':
    main()
