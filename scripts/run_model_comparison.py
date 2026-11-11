"""按统一数据契约构建矩阵、训练基线、冻结选择并生成研究结论。"""

import argparse
from pathlib import Path

from mm_research.models.comparison import prepare_comparison, train_comparison_linear, train_comparison_tree
from mm_research.models.comparison import train_comparison_gru, freeze_comparison, evaluate_comparison_family
from mm_research.models.comparison_report import write_comparison_report


# 【非核心】同一配置显式指定执行阶段，各阶段由受控队列独立运行。
def main() -> None:
    """输入模型对照YAML及阶段，可选具名输入组或GRU历史秒数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('phase', choices=['prepare', 'linear', 'tree', 'gru', 'freeze', 'evaluate_cpu', 'evaluate_gpu', 'report'])
    parser.add_argument('--group')
    parser.add_argument('--model')
    parser.add_argument('--sequence', type=int)
    arguments = parser.parse_args()
    if arguments.phase == 'prepare':
        prepare_comparison(arguments.config)
    elif arguments.phase == 'linear':
        train_comparison_linear(arguments.config)
    elif arguments.phase == 'tree':
        train_comparison_tree(arguments.config, arguments.group)
    elif arguments.phase == 'gru':
        train_comparison_gru(arguments.config, arguments.sequence)
    elif arguments.phase == 'freeze':
        freeze_comparison(arguments.config)
    elif arguments.phase == 'report':
        write_comparison_report(arguments.config)
    else:
        evaluate_comparison_family(arguments.config, arguments.phase == 'evaluate_gpu', arguments.model)


if __name__ == '__main__':
    main()
