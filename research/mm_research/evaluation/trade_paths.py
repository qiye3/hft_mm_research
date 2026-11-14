"""开发期入场后的价格路径诊断；未来路径仅作归因，不作为交易输入。"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.multihorizon.common import SECOND, context


# 【核心逻辑418】以入场所在秒Mid为基准，计算完整有效路径的方向收益与极值。
def price_path(market: np.ndarray, entry_row: int, horizon: int, side: float) -> dict:
    """跨质量区段或无效秒不拼接；极值包含起点，且不是可实现的择时收益。"""
    window = market[entry_row:entry_row + horizon + 1]
    complete = len(window) == horizon + 1
    valid = complete and bool(np.all(
        (window[:, 8] > .5) & (window[:, 9] == window[0, 9])
        & np.isfinite(window[:, 0]) & (window[:, 0] > 0)
    ))
    result = {
        'path_valid': valid,
        'mid_return_bps': None,
        'mfe_bps': None,
        'mae_bps': None,
        'mfe_time_s': None,
        'mae_time_s': None,
    }
    if not valid:
        return result
    returns = side * np.log(window[:, 0] / window[0, 0]) * 10000
    result.update({
        'mid_return_bps': float(returns[-1]),
        'mfe_bps': float(returns.max()),
        'mae_bps': float(returns.min()),
        'mfe_time_s': int(returns.argmax()),
        'mae_time_s': int(returns.argmin()),
    })
    return result


# 【核心逻辑419】按模型、实际盈亏及期限汇总入场后路径，保留缺口样本计数。
def evaluate_trade_paths(filename: Path) -> None:
    """复用已有开发账本；固定期限路径不含交易费用，也不冒充策略回测。"""
    _, config, output = context(filename)
    source = Path(config['source']) / 'e0'
    manifest = json.loads((source / 'manifest.json').read_text())
    bounds = json.loads((source / 'splits.json').read_text())['segments']['val_dev']
    market = np.load(source / 'market.npy', mmap_mode='r')[:bounds['stop_row']]
    trades = pl.read_parquet(output / 'trades_with_regime.parquet')
    records = []
    for trade in trades.iter_rows(named=True):
        entry_row = round(trade['entry_time_s'] - manifest['start_timestamp_ns'] / SECOND - .002)
        for horizon in [30, 60, 180, 600, 1800, 3600]:
            record = {
                'model': trade['model'],
                'trade_id': trade['trade_id'],
                'horizon_s': horizon,
                'actual_profitable': trade['net_pnl_usdt'] > 0,
                'actual_net_usdt': trade['net_pnl_usdt'],
                'trend': trade['trend'],
                'holding_seconds': trade['holding_seconds'],
            }
            record.update(price_path(market, entry_row, horizon, trade['side']))
            records.append(record)
    paths = pl.DataFrame(records)
    paths.write_parquet(output / 'trade_paths.parquet')
    summary = paths.group_by(['model', 'horizon_s', 'actual_profitable']).agg(
        pl.len().alias('trades'),
        pl.col('path_valid').sum().alias('valid_paths'),
        pl.col('mid_return_bps').mean().alias('mean_mid_bps'),
        pl.col('mid_return_bps').median().alias('median_mid_bps'),
        pl.col('mfe_bps').mean().alias('mean_mfe_bps'),
        pl.col('mae_bps').mean().alias('mean_mae_bps'),
        pl.col('mfe_time_s').median().alias('median_mfe_time_s'),
    ).sort(['model', 'horizon_s', 'actual_profitable'])
    summary.write_csv(output / 'trade_path_summary.csv')
