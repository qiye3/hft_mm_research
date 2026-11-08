"""只读研究Parquet，输出看板自己的紧凑诊断；不写主评估目录。"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import fcntl
import json
from pathlib import Path

import numpy as np
import polars as pl

from .statistics import QUANTILES, SCOPES, group_statistics, pearson

APP = Path(__file__).resolve().parents[2]
PROJECT = APP.parents[1]
FACTORS = PROJECT / 'workspace/data/factors/gen001_v1'
FEATURES = PROJECT / 'workspace/data/features/market_columns_v1'
LABELS = PROJECT / 'workspace/data/labels/market_mid_columns_v1'
RESULTS = PROJECT / 'workspace/results/evaluation/gen001_v1'
KEYS = ['symbol', 'date', 'timestamp_ns']
SHORT_NAMES = {
    'book_obi_l1_mean_diff_10_50': '一档盘口压力',
    'book_obi_l5_mean_diff_10_50': '五档盘口压力',
    'book_obi_l10_mean_diff_10_50': '十档盘口压力',
    'trade_aggr_imb_100ms_mean_diff_10_50': '100ms成交压力',
    'trade_aggr_imb_1s_mean_diff_10_50': '1s成交压力',
    'cross_trade_book_gap_mean_diff_10_50': '成交与盘口背离',
}


# 【非核心】复用已经存在的项目锁路径，仅以只读方式打开，不创建或改写主目录。
@contextmanager
def source_locks(paths: list[Path]):
    """在物化查询期间持有共享锁；来源必须是项目已经发布的文件。"""
    with ExitStack() as descriptors:
        for path in sorted(set(paths)):
            descriptor = descriptors.enter_context((path.parent / '.locks' / f'{path.name}.lock').open('rb'))
            fcntl.flock(descriptor.fileno(), fcntl.LOCK_SH)
        yield


# 【非核心】从单因子全日期文件读取元信息，保留已有公式和时钟身份。
def metadata(path: Path) -> dict:
    """输入项目Parquet，返回mm_research元信息。"""
    return json.loads(pl.read_parquet_metadata(path)['mm_research'])


# 【非核心】首尾精确按时间键关联，历史收益只读取过去，缺失网格不拼接。
def time_series(name: str, date: str, start_seconds: int, duration_seconds: int,
                horizon_ms: int, expected_metadata: dict) -> dict:
    """返回最多6000个100ms网格，含因子、未来标签、过去1s收益和快照年龄。"""
    suffix = f'{horizon_ms}ms'
    if horizon_ms % 1000 == 0:
        suffix = f'{horizon_ms // 1000}s'
    factor_path = FACTORS / f'{name}.parquet'
    label_path = LABELS / f'horizon={suffix}' / f'{date}.parquet'
    feature_paths = [FEATURES / f'{column}.parquet' for column in ['return_1s_bps', 'snapshot_age_ns']]
    start = int(np.datetime64(date, 'ns').astype(np.int64)) + start_seconds * 1_000_000_000
    end = start + duration_seconds * 1_000_000_000
    with source_locks([factor_path, label_path, *feature_paths]):
        if metadata(factor_path) != expected_metadata:
            raise ValueError('因子来源已改变，请重新prepare看板快照')
        for path in [label_path, *feature_paths]:
            if metadata(path)['dataset'] != expected_metadata['dataset']:
                raise ValueError('时序查询来源与看板快照不一致')
        values = pl.scan_parquet(factor_path).filter(pl.col('date') == date).filter(
            pl.col('timestamp_ns').is_between(start, end, closed='left'),
        ).select(*KEYS, pl.col(name).alias('factor'))
        labels = pl.scan_parquet(label_path).select(
            *KEYS, pl.when(pl.col('label_valid')).then(pl.col(f'y_mid_{suffix}_bps')).alias('label'),
        )
        joined = values.join(labels, on=KEYS, how='left', validate='1:1')
        for column, path in zip(['return_1s_bps', 'snapshot_age_ns'], feature_paths):
            joined = joined.join(pl.scan_parquet(path).select(*KEYS, column), on=KEYS,
                                 how='left', validate='1:1')
        rows = joined.sort('timestamp_ns').select(
            pl.col('timestamp_ns').alias('ts'), 'factor', 'label',
            pl.col('return_1s_bps').alias('past'),
            (pl.col('snapshot_age_ns') / 1_000_000).alias('age_ms'),
        ).collect().to_dicts()
    return {'date': date, 'horizon_ms': horizon_ms, 'series': rows}


# 【非核心】标签沿用现有文件；十二组只做看板扩展，不重算因子。
def evaluate_day(values: pl.DataFrame, date: str, fixed_thresholds: list[float],
                 original_ic: pl.DataFrame, expected_dataset: dict) -> dict:
    """输入一个因子日，返回四周期覆盖、两种十二组及信号稳定性诊断。"""
    finite = values.filter(pl.col('value').is_finite())['value']
    thresholds = [finite.quantile(quantile, interpolation='linear') for quantile in QUANTILES]
    groups = {'daily': {}, 'fixed': {}}
    coverage = {}
    for horizon_ms, suffix in [(100, '100ms'), (500, '500ms'), (1000, '1s'), (5000, '5s')]:
        path = LABELS / f'horizon={suffix}' / f'{date}.parquet'
        with source_locks([path]):
            target_metadata = metadata(path)
            labels = pl.read_parquet(path).select(
                *KEYS, 'label_valid', pl.col(f'y_mid_{suffix}_bps').alias('target'),
            )
        if target_metadata['dataset'] != expected_dataset:
            raise ValueError('因子与标签数据版本不一致')
        if target_metadata['horizon_ns'] != horizon_ms * 1_000_000:
            raise ValueError('标签周期与路径不符')
        joined = values.join(labels, on=KEYS, how='left', validate='1:1')
        valid_factor = pl.col('value').is_finite().fill_null(False)
        valid_target = (pl.col('label_valid').fill_null(False) & pl.col('target').is_finite()).fill_null(False)
        coverage[str(horizon_ms)] = joined.select(
            pl.len().alias('rows'), valid_factor.sum().alias('factor_valid'),
            valid_target.sum().alias('label_valid'),
            (valid_factor & valid_target).sum().alias('paired'),
            (~valid_factor).sum().alias('factor_invalid'),
            (valid_factor & ~valid_target).sum().alias('label_invalid_only'),
            (valid_target & (pl.col('target') == 0)).sum().alias('zero_labels'),
        ).row(0, named=True)
        for mode, cutoffs in [('daily', thresholds), ('fixed', fixed_thresholds)]:
            groups[mode][str(horizon_ms)] = group_statistics(joined, cutoffs)
    distribution = values.select(
        *[pl.col('value').quantile(quantile, interpolation='linear').alias(key)
          for key, quantile in [('p10', .1), ('p25', .25), ('p50', .5), ('p75', .75), ('p90', .9)]],
    ).row(0, named=True)
    ic_rows = original_ic.filter(pl.col('date') == date).select(
        (pl.col('horizon_ns') // 1_000_000).alias('h'), 'ic', 'rank_ic',
    ).to_dicts()
    return {'date': date, 'groups': groups, 'coverage': coverage, 'ic': ic_rows,
            'percentiles': distribution, 'daily_thresholds': thresholds}


# 【非核心】固定h=1s，等待δ后评估剩余信息；不把等待期间涨跌计入目标。
def delay_diagnostics(values: pl.DataFrame, date: str) -> list[dict]:
    """以当前因子与未来δ时刻已经生成的1s标签关联，保持同日有效性。"""
    path = LABELS / 'horizon=1s' / f'{date}.parquet'
    with source_locks([path]):
        labels = pl.read_parquet(path).select('timestamp_ns', 'label_valid', 'y_mid_1s_bps')
    output = []
    for delay_ms in [0, 100, 200, 500, 1000]:
        shifted = labels.with_columns((pl.col('timestamp_ns') - delay_ms * 1_000_000).alias('timestamp_ns'))
        paired = values.join(shifted, on='timestamp_ns', how='left').filter(
            pl.col('value').is_finite() & pl.col('label_valid') & pl.col('y_mid_1s_bps').is_finite(),
        )
        output.append({'day': date, 'delay_ms': delay_ms, 'count': paired.height,
                       'ic': pearson(paired['value'].to_numpy(), paired['y_mid_1s_bps'].to_numpy())})
    return output


# 【非核心】按全期选定因子的共同有限样本计算精确相关，逐日结果支持范围聚合。
def correlation_days(names: list[str], dates: list[str]) -> list[dict]:
    """输出六因子共同样本的日相关、计数和用于全期Pearson的充分统计量。"""
    paths = [FACTORS / f'{name}.parquet' for name in names]
    rows = []
    with source_locks(paths):
        for date in dates:
            scans = [pl.scan_parquet(path).filter(pl.col('date') == date) for path in paths]
            joined = scans[0]
            for scan in scans[1:]:
                joined = joined.join(scan, on=KEYS, how='inner', validate='1:1')
            common = joined.filter(pl.all_horizontal([pl.col(name).is_finite() for name in names])).collect()
            matrix = common.select(names).to_numpy()
            ranks = common.select([pl.col(name).rank(method='average') for name in names]).to_numpy()
            pearson_matrix = np.corrcoef(matrix, rowvar=False)
            rank_matrix = np.corrcoef(ranks, rowvar=False)
            sums = matrix.sum(axis=0)
            products = matrix.T @ matrix
            for left_index, left in enumerate(names):
                for right_index, right in enumerate(names):
                    rows.append({
                        'date': date, 'left': left, 'right': right, 'count': common.height,
                        'sum_x': float(sums[left_index]), 'sum_y': float(sums[right_index]),
                        'sum_x2': float(products[left_index, left_index]),
                        'sum_y2': float(products[right_index, right_index]),
                        'sum_xy': float(products[left_index, right_index]),
                        'pearson': float(pearson_matrix[left_index, right_index]),
                        'spearman': float(rank_matrix[left_index, right_index]),
                    })
            print(f'相关性 {date}', flush=True)
    return rows
