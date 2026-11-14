"""在同一扩展因子池中比较家族覆盖与相关性去冗余，选择只读取内部拟合期。"""

from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import context


# 【核心逻辑409】以同一128列宽池比较选列规则，输出列索引，模型直接复用原缓存。
def select_feature_pool(filename: Path) -> None:
    """家族模式限制同类占比，去冗余模式保留低重复列；不预设越来越小的目标数量。"""
    _, config, output = context(filename)
    source = Path(config['inputs'])
    candidates = pl.read_csv(source / 'selection.csv').with_row_index('column_index')
    inventory = pl.read_csv(Path(config['factor_results']) / 'summary/factor_inventory.csv')
    candidates = candidates.join(inventory.select('factor', 'batch'), on='factor', how='left')
    candidates = candidates.sort(['score', 'factor'], descending=[True, False])
    if config['method'] == 'family_balanced':
        chosen = candidates.group_by('batch', maintain_order=True).head(config['maximum_per_family'])
    elif config['method'] == 'correlation_pruned':
        fit_rows = np.load(Path(config['source']) / 'e1/folds/final/fit_rows.npy')
        sampled = np.flatnonzero(fit_rows % config['sample_interval_s'] == 0)
        matrix = np.load(source / 'fit_x.npy', mmap_mode='r')
        count = candidates.height
        values = np.asarray(matrix[sampled, :count], dtype='float64')
        values[matrix[sampled, count:] > .5] = np.nan
        correlations = pairwise_correlation(values, config['minimum_pairs'])
        selected = []
        for row in candidates.iter_rows(named=True):
            index = row['column_index']
            if all(abs(correlations[index, previous]) < config['maximum_correlation'] for previous in selected):
                selected.append(index)
        chosen = candidates.filter(pl.col('column_index').is_in(selected))
        np.save(output / 'fit_correlations.npy', correlations)
    else:
        raise ValueError(f"未知因子池筛选规则：{config['method']}")
    indices = chosen['column_index'].to_list()
    chosen.write_csv(output / 'selection.csv')
    write_json_atomic(output / 'selected.json', {
        'method': config['method'],
        'factors': chosen['factor'].to_list(),
        'matrix_columns': indices + [index + candidates.height for index in indices],
        'source_columns': candidates.height,
        'selected_columns': len(indices),
        'selection_scope': 'internal_fit_only',
    })


# 【核心逻辑410】按两列同时有效的拟合样本计算Pearson相关，不把缺失填补值计为真实观测。
def pairwise_correlation(values: np.ndarray, minimum_pairs: int) -> np.ndarray:
    """矩阵乘法汇总成对计数与一二阶矩；样本不足时明确失败，不静默认为不相关。"""
    observed = np.isfinite(values).astype('float64')
    filled = np.where(np.isfinite(values), values, 0.)
    counts = observed.T @ observed
    if np.any(counts < minimum_pairs):
        raise ValueError('拟合期成对有效样本不足，无法执行相关性去冗余')
    sums = filled.T @ observed
    squares = (filled * filled).T @ observed
    covariance = filled.T @ filled - sums * sums.T / counts
    variance = squares - sums * sums / counts
    denominator = np.sqrt(np.maximum(variance, 0.) * np.maximum(variance.T, 0.))
    if np.any(denominator == 0):
        raise ValueError('相关性候选含常数列，需先检查因子定义')
    return np.clip(covariance / denominator, -1., 1.)
