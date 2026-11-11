"""模型盘点、冻结信号、原生策略回放及账本分析的薄入口。"""

import argparse
from pathlib import Path

import yaml


# 【非核心】显式选择一个阶段，所属配置内声明输入、输出和费用假设。
def main():
    """每次执行一个阶段，研究核心逻辑分别归属Models和Strategies包。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['inventory', 'signals', 'sweep', 'report'])
    arguments = parser.parse_args()
    root = Path.cwd()
    config = yaml.safe_load(arguments.config.read_text())
    output = root / config['output']
    output.mkdir(parents=True, exist_ok=True)
    if arguments.stage == 'inventory':
        from mm_research.models.inventory import collect_model_inventory
        collect_model_inventory(root, output)
    elif arguments.stage == 'signals':
        from mm_research.strategies.signals import export_prediction_signals
        export_prediction_signals(root, output, config['date'])
    elif arguments.stage == 'sweep':
        from mm_research.strategies.experiments import run_strategy_sweep
        run_strategy_sweep(root, config)
    else:
        from mm_research.strategies.analysis import build_strategy_report
        build_strategy_report(root, config)


if __name__ == '__main__':
    main()
