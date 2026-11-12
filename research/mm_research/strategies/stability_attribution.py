"""对已冻结策略做波动环境与方向归因，不参与策略选择。"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.multihorizon.common import context, SECOND
from mm_research.strategies.cost_attribution import past_volatility


# 【核心逻辑388】沿用上一轮Train波动分界，逐笔归因所有完成案例，保留缺失环境类别。
def attribute_stability(filename: Path) -> None:
    """输出方向/波动分组、持仓时长和收益集中度，不过滤亏损或未知状态交易。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    definition = Path('workspace/results/cost_attribution_20261111_v1/volatility_definition.json')
    thresholds = json.loads(definition.read_text())['thresholds_bps']
    origin = manifest['start_timestamp_ns'] / SECOND
    summaries = []
    for file in sorted((output / 'cases').glob('*.trades.parquet')):
        identifier = file.name.removesuffix('.trades.parquet')
        trades = pl.read_parquet(file)
        rows = np.floor(trades['entry_time_s'].to_numpy() - origin).astype('int64')
        volatility = past_volatility(market, rows)
        regime = np.full(len(rows), 'unknown', dtype=object)
        finite = np.isfinite(volatility)
        regime[finite & (volatility < thresholds[0])] = 'low'
        regime[finite & (volatility >= thresholds[0]) & (volatility < thresholds[1])] = 'medium'
        regime[finite & (volatility >= thresholds[1])] = 'high'
        trades = trades.with_columns(
            pl.Series('past600_vol_bps', volatility), pl.Series('volatility_regime', regime.tolist(), dtype=pl.String),
            (pl.col('exit_time_s') - pl.col('entry_time_s')).alias('holding_seconds'),
        )
        trades.write_parquet(output / 'cases' / f'{identifier}.attribution.parquet', compression='zstd')
        for group in ['volatility_regime', 'side']:
            summary = trades.group_by(group).agg(
                pl.len().alias('trades'), pl.col('gross_pnl_usdt').sum(), pl.col('fees_usdt').sum(),
                pl.col('funding_paid_usdt').sum(), pl.col('net_pnl_usdt').sum(),
                pl.col('net_pnl_usdt').min().alias('worst_trade_usdt'),
                pl.col('holding_seconds').mean(),
            )
            for row in summary.iter_rows(named=True):
                summaries.append({'case': identifier, 'dimension': group, 'category': str(row.pop(group)), **row})
    pl.DataFrame(summaries).write_csv(output / 'regime_attribution.csv')
