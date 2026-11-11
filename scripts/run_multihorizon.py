"""多期限研究的薄入口，源码快照也可从服务器项目根目录独立运行。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))


# 【非核心】显式指定研究配置、阶段和模型任务，数据根目录沿用调用时工作目录。
def main():
    """调度包内实现，所有配置和研究结果归属独立版本。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'persistence', 'inventory', 'opportunities', 'baselines', 'execution', 'report', 'night'])
    parser.add_argument('--model', default='M1')
    parser.add_argument('--fold', default='final')
    arguments = parser.parse_args()
    if arguments.stage == 'prepare':
        from mm_research.multihorizon.data import prepare_dataset
        prepare_dataset(arguments.config)
    elif arguments.stage == 'train':
        from mm_research.multihorizon.models import train_model
        train_model(arguments.config, arguments.model, arguments.fold)
    elif arguments.stage == 'evaluate':
        from mm_research.multihorizon.evaluation import evaluate_models
        evaluate_models(arguments.config)
    elif arguments.stage == 'persistence':
        from mm_research.multihorizon.persistence import analyze_persistence
        analyze_persistence(arguments.config)
    elif arguments.stage == 'inventory':
        from mm_research.multihorizon.inventory import run_inventory
        run_inventory(arguments.config)
    elif arguments.stage == 'execution':
        from mm_research.multihorizon.execution import run_execution
        run_execution(arguments.config)
    elif arguments.stage == 'opportunities':
        from mm_research.multihorizon.execution import run_training_opportunities
        run_training_opportunities(arguments.config)
    elif arguments.stage == 'baselines':
        from mm_research.multihorizon.execution import run_training_opportunities
        run_training_opportunities(arguments.config, baseline_only=True)
    elif arguments.stage == 'night':
        from mm_research.multihorizon.night import run_night
        run_night(arguments.config)
    else:
        from mm_research.multihorizon.report import write_report
        write_report(arguments.config)


if __name__ == '__main__':
    main()
