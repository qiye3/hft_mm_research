"""从标准官方逐笔构造桶内规模分布，发布九个全日期100ms Features供Gen023使用。"""

import datetime as dt
from pathlib import Path

import polars as pl

from mm_research.data.sources import DAY_NS, scan_standard_history
from mm_research.data.storage import artifact_ready, file_locks, read_metadata, scan_series, write_series_frames
from mm_research.execution import ordered_process_map
from mm_research.features.trade_size import INVALID_FLAGS

DISTRIBUTION_COLUMNS = [f'trade_size_{side}_{metric}_100ms' for side in ['buy', 'sell', 'all']
                        for metric in ['median_qty', 'topdecile_share', 'hhi']]


# 【核心逻辑31】在同一100ms桶内分别计算买、卖、全部逐笔规模的三个分布统计。
def distribution_bins(trades: pl.LazyFrame) -> pl.LazyFrame:
    """输入已过滤合法成交及timestamp_ns，零成交子组为零，头部取ceil(10%笔数)。"""
    expressions = []
    for side in ['buy', 'sell', 'all']:
        quantity = pl.col('quantity') if side == 'all' else pl.col('quantity').filter(pl.col('side') == side)
        count = quantity.len()
        total = quantity.sum()
        top_count = (count.cast(pl.Float64) * .1).ceil().cast(pl.Int64)
        top = quantity.sort(descending=True).head(top_count).sum()
        expressions.extend([
            pl.when(count > 0).then(quantity.median()).otherwise(0).alias(f'trade_size_{side}_median_qty_100ms'),
            pl.when(count > 0).then(top / total).otherwise(0).alias(f'trade_size_{side}_topdecile_share_100ms'),
            pl.when(count > 0).then(quantity.pow(2).sum() / total.pow(2)).otherwise(0).alias(f'trade_size_{side}_hhi_100ms'),
        ])
    return trades.group_by('timestamp_ns').agg(expressions)


# 【核心逻辑30】复用标准成交与既有网格质量，对桶内分布进行时钟对齐。
def distribution_day(task: tuple) -> pl.DataFrame:
    """未知桶置空，已知空桶置零，关联保留100ms时间顺序。"""
    standard, features, date, dataset = task
    interval = dataset['interval_ns']
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    source = scan_standard_history(standard, 'trades', date).filter(
        ~pl.col('quality_flags').list.eval(pl.element().is_in(INVALID_FLAGS)).list.any(),
    ).with_columns(
        ((pl.col('exchange_timestamp_ns') + dataset['market_data_latency_ns'] + interval - 1) // interval * interval).alias('timestamp_ns'),
    ).filter(pl.col('timestamp_ns').is_between(start, start + DAY_NS, closed='left'))
    bins = distribution_bins(source)
    quality, _ = scan_series(features, ['flow_valid'], [date])
    result = quality.join(bins, on='timestamp_ns', how='left', maintain_order='left').with_columns([
        pl.when(pl.col('flow_valid')).then(pl.col(name).fill_null(0)).otherwise(None).cast(pl.Float64).alias(name)
        for name in DISTRIBUTION_COLUMNS
    ])
    return result.collect(engine='streaming')


# 【核心逻辑29】按日期有界计算交易分布，原子发布九个全日期特征。
def build_distribution_features(root: Path, config: dict, workers: int) -> None:
    """输入既有Features配置和并发数，复用相同定义已有列，不重算其他Features。"""
    features = root / config['feature_root'] / config['feature_version']
    standard = root / config['standard_root'] / config['standard_version']
    dataset = read_metadata(features / 'flow_valid.parquet')['dataset']
    metadata = {name: {'kind': 'feature', 'name': name, 'dataset': dataset,
                       'definition': {'group': 'trade_distribution', 'version': 'v1', 'top_fraction': .1}}
                for name in DISTRIBUTION_COLUMNS}
    inputs = [standard / 'trades' / f'{date}.parquet' for date in dataset['dates']]
    inputs.append(features / 'flow_valid.parquet')
    with file_locks(inputs, shared=True), file_locks([features / f'{name}.parquet' for name in DISTRIBUTION_COLUMNS]):
        pending = [name for name in DISTRIBUTION_COLUMNS if not artifact_ready(features / f'{name}.parquet', metadata[name], False)]
        if pending:
            tasks = [(standard, features, date, dataset) for date in dataset['dates']]
            write_series_frames(ordered_process_map(distribution_day, tasks, workers), features, pending, metadata)
    print('逐笔规模分布Features完成', flush=True)
