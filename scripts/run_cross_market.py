"""跨市场信息增量实验入口，数据计算直接服务本批模型训练。"""

import argparse
from pathlib import Path

from mm_research.models.cross_market import (
    cache_group, prepare_cross_inputs, prepare_old_cross_inputs, report_cross_models, train_cross_model,
)


# 【非核心】命令只分发已有研究模块，公式和训练主体保留在对应模块。
def main() -> None:
    """执行输入计算、分组缓存、旧输入复用、GPU训练或结果汇总。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['inputs', 'cache', 'old', 'train', 'report', 'ablation_inputs', 'ablation_report'])
    parser.add_argument('--group', choices=['spot', 'premium', 'all', 'flow', 'price', 'baseline'], default='all')
    parser.add_argument('--family', choices=['xgb', 'mlp'], default='xgb')
    parser.add_argument('--seed', type=int, default=17)
    arguments = parser.parse_args()
    if arguments.stage == 'inputs':
        prepare_cross_inputs(arguments.config)
    elif arguments.stage == 'cache':
        cache_group(arguments.config, arguments.group)
    elif arguments.stage == 'old':
        prepare_old_cross_inputs(arguments.config)
    elif arguments.stage == 'train':
        train_cross_model(arguments.config, arguments.group, arguments.family, arguments.seed)
    elif arguments.stage == 'report':
        report_cross_models(arguments.config)
    else:
        from mm_research.models.cross_market_ablation import prepare_ablation, report_ablation

        if arguments.stage == 'ablation_inputs':
            prepare_ablation(arguments.config)
        else:
            report_ablation(arguments.config)


if __name__ == '__main__':
    main()
