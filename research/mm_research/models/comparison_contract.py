"""正式模型对照的数据契约，选列仅读取训练期评价，样本和缺失规则对全部实验固定。"""

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.factors.dataset import factor_module
from mm_research.models.dataset import RAW_COLUMNS


# 【核心逻辑134】从训练期IC和覆盖率确定候选、代表列、深度扩池及序列输入。
def select_comparison_inputs(root: Path, config: dict) -> dict:
    """返回输入目录、候选组和派生定义；验证及测试统计不参与筛选。"""
    factor_parameters = {
        "min_fraction": .8,
        "tick_size": .1,
        "aggregation_windows_s": [10, 60, 180, 600],
        "mean_diff_windows_s": [[10, 60], [60, 180], [180, 600]],
    }
    records = []
    locations = {}
    batches = {}
    families = {}
    for family, numbers in config['families'].items():
        for number in numbers:
            batch = f'gen{number:03d}'
            source = next(source for source in config['factor_sources'] if batch in source['batches'])
            names = factor_module(batch).factor_expressions(factor_parameters)
            for name in names:
                locations[name] = source['folder']
                batches[name] = batch
                families[name] = family
                filename = root / source['evaluation'] / name / 'summary.parquet'
                rows = pl.scan_parquet(filename).filter(
                    (pl.col('split') == 'train') & (pl.col('sampling') == 'all')
                    & (pl.col('horizon_ns') == 1_000_000_000)
                ).collect().to_dicts()
                records.extend(rows)
    days = (dt.date.fromisoformat(config['train_end']) - dt.date.fromisoformat(config['start_date'])).days + 1
    ranked = pl.DataFrame(records).filter(
        pl.col('ic_mean').is_finite()
        & (pl.col('valid_ic_days') >= int(np.ceil(days * .9)))
        & (pl.col('paired_rows') >= config['minimum_train_coverage'] * days * 86400)
    ).sort([pl.col('ic_mean').abs(), pl.col('factor')], descending=[True, False])
    order = ranked['factor'].to_list()
    representatives = []
    for family in config['families']:
        representatives.extend([name for name in order if families[name] == family][:2])
    representatives = list(dict.fromkeys([*representatives, *order]))[:16]
    anchors = []
    for family in config['families']:
        anchors.extend([name for name in order if families[name] == family][:1])
    anchors = list(dict.fromkeys([*anchors, *representatives]))[:10]
    base = [name for name in order if name not in anchors][:80] + anchors
    if len(base) != 90 or len(representatives) != 16:
        raise ValueError('训练期覆盖合格的候选不足以完成预设90列和16列对照')
    extra = [name for name in order if families[name] == 'depth' and name not in base][:40]
    sequence_sources = list(dict.fromkeys([*anchors, *representatives]))[:12]
    derivations = {}
    ratio_inputs = []
    for depth in [1, 5, 10]:
        for window in [5, 10, 50]:
            bid = f'book_bid_depth_l{depth}_delta_100ms_sum_{window}s'
            ask = f'book_ask_depth_l{depth}_delta_100ms_sum_{window}s'
            ratio_inputs.extend([bid, ask])
            derivations[f'depth_ratio_l{depth}_{window}s'] = {'kind': 'ratio', 'inputs': [bid, ask]}
    ratios = list(derivations)
    for name in sequence_sources:
        for seconds in [1, 5, 30]:
            derivations[f'{name}__lag_{seconds}s'] = {'kind': 'lag', 'input': name, 'seconds': seconds}
        for seconds in [30, 120]:
            derivations[f'{name}__deviation_{seconds}s'] = {'kind': 'deviation', 'input': name, 'seconds': seconds}
    sequences = [name for name in derivations if name not in ratios]
    full = list(dict.fromkeys([*base, *extra, *ratios, *sequences]))
    groups = {
        'history': [name for name in order if families[name] == 'price'][:2],
        'core': [name for name in order if families[name] == 'obi'][:1]
                + [name for name in order if batches[name] == 'gen006'][:1],
        'representatives16': representatives, 'base90': base,
        'depth_expanded': list(dict.fromkeys([*base, *extra, *ratios])),
        'sequence_expanded': [*base, *sequences], 'full': full,
        'without_depth': [name for name in full if name not in ratios and families[name.split('__')[0]] != 'depth'],
        'without_obi': [name for name in full if name in ratios or families[name.split('__')[0]] != 'obi'],
    }
    groups['core'] = list(dict.fromkeys([*groups['history'], *groups['core']]))
    required = list(dict.fromkeys([*RAW_COLUMNS, *base, *extra, *ratio_inputs, *representatives]))
    inputs = {name: config['feature_folder'] if name in RAW_COLUMNS else locations[name] for name in required}
    return {'inputs': inputs, 'derivations': derivations, 'groups': groups,
            'family_by_factor': families, 'batch_by_factor': batches,
            'training_ranking': ranked.to_dicts(), 'criterion': 'train_only_1s_daily_ic_and_train_coverage',
            'selection_uses_validation': False, 'selection_uses_test': False}


# 【核心逻辑137】在整秒输入上构造有界历史派生量，缺失保持时间位置并跨午夜连续。
def comparison_derivations(specifications: dict) -> list[pl.Expr]:
    """比例分母下限1e-6，滞后和含当前的滚动均值偏离仅使用当时已知数据。"""
    expressions = []
    for name, specification in specifications.items():
        kind = specification['kind']
        if kind == 'ratio':
            bid, ask = [pl.col(column) for column in specification['inputs']]
            values = (bid - ask) / (bid.abs() + ask.abs()).clip(lower_bound=1e-6)
        elif kind == 'lag':
            values = pl.col(specification['input']).shift(specification['seconds'])
        elif kind == 'deviation':
            window = specification['seconds']
            column = pl.col(specification['input'])
            values = column - column.rolling_mean(window, min_samples=int(np.ceil(window * .8)))
        else:
            raise ValueError(f'未知模型派生量 {kind}')
        expressions.append(values.cast(pl.Float64).alias(name))
    return expressions


# 【核心逻辑138】从共同有效标签确定所有模型共用的训练抽样、调参位置及完整评估样本。
def comparison_indices(metadata: dict, targets, config: dict) -> dict[str, np.ndarray]:
    """首个交易日预热120秒；边界300秒隔离及目标时间检查已在标签缓存实施。"""
    ranges = {'train': (120, metadata['train_stop']),
              'validation': (metadata['validation_start'], metadata['validation_stop']),
              'test': (metadata['test_start'], metadata['rows'])}
    result = {}
    for split, (start, stop) in ranges.items():
        indices = np.arange(start, stop, dtype=np.int64)
        result[split] = indices[np.isfinite(targets[indices, 0])]
    generator = np.random.default_rng(config['seed'])
    result['fit'] = np.sort(generator.choice(result['train'], min(config['training_rows'], len(result['train'])), replace=False))
    result['tune'] = result['validation'][np.linspace(0, len(result['validation']) - 1, min(config['validation_rows'], len(result['validation'])), dtype=int)]
    result['threshold'] = result['fit'][np.linspace(0, len(result['fit']) - 1, min(100000, len(result['fit'])), dtype=int)]
    return result
