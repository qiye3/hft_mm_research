"""四期限新旧输入模型对照入口，独立任务交给既有后台队列托管。"""

import argparse
from pathlib import Path

from mm_research.models.input_comparison import (
    prepare_new_inputs, prepare_old_inputs, reuse_old180, summarize_input_models, train_input_group,
)


# 【非核心】显式选择缓存准备、模型训练或汇总阶段，所有产物由模型配置管理。
def main() -> None:
    """解析本轮阶段、预测周期、输入组、模型和种子。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    stages = parser.add_subparsers(dest='stage', required=True)
    stages.add_parser('prepare-old')
    prepare = stages.add_parser('prepare-new')
    prepare.add_argument('--horizon', type=int, required=True, choices=[30, 60, 120, 180])
    train = stages.add_parser('train')
    train.add_argument('--horizon', type=int, required=True, choices=[30, 60, 120, 180])
    train.add_argument('--group', required=True, choices=['old', 'new', 'combined'])
    train.add_argument('--family', required=True, choices=['mlp', 'xgb'])
    train.add_argument('--seed', type=int, required=True)
    reuse = stages.add_parser('reuse-old180')
    reuse.add_argument('--family', required=True, choices=['mlp', 'xgb'])
    reuse.add_argument('--seed', type=int, required=True)
    reuse.add_argument('--from-folder', type=Path, required=True)
    stages.add_parser('summary')
    arguments = parser.parse_args()
    if arguments.stage == 'prepare-old':
        prepare_old_inputs(arguments.config)
    elif arguments.stage == 'prepare-new':
        prepare_new_inputs(arguments.config, arguments.horizon)
    elif arguments.stage == 'train':
        train_input_group(arguments.config, arguments.horizon, arguments.group, arguments.family, arguments.seed)
    elif arguments.stage == 'reuse-old180':
        reuse_old180(arguments.config, arguments.family, arguments.seed, arguments.from_folder)
    else:
        summarize_input_models(arguments.config)


if __name__ == '__main__':
    main()
