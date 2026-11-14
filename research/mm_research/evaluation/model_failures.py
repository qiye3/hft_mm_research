"""对齐费用模型输出、固定期限标签与真实交易账本，保留完整开发时间轴。"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import context, SECOND


# 【核心逻辑411】逐模型输出分布、强度分组、时间相关和实际成交失败样本，仅分析开发期。
def evaluate_model_failures(filename: Path) -> None:
    """固定期限标签与实际持有收益分别保存，分钟价格/信号和账户权益用时间键关联。"""
    from mm_research.models.cost_classifier import net_labels
    from mm_research.strategies.stable_strategy import visible_management

    _, config, output = context(filename)
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    bounds = json.loads((source / 'e0/splits.json').read_text())['segments']['val_dev']
    rows = np.arange(bounds['start_row'], bounds['stop_row'])
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    timestamps = manifest['start_timestamp_ns'] + rows * SECOND
    risk = visible_management(market, rows[0], rows[-1] + 1)[:, 2]
    profiles, bins, correlations, daily_frames, trade_frames = [], [], [], [], []
    for name, specification in config['models'].items():
        timeline = pl.DataFrame({'timestamp_ns': timestamps[::60], 'mid': market[rows[::60], 0]})
        side_predictions = []
        for side in [1, -1]:
            folder = Path(specification['folder']) / 'models' / f'h1800_side{side}'
            metadata = json.loads((folder / 'completed.json').read_text())
            edge = np.load(folder / 'val_dev_net_edge_bps.npy', mmap_mode='r')
            side_predictions.append(edge)
            probability = (edge - metadata['train_loss_mean_bps']) / (
                metadata['train_win_mean_bps'] - metadata['train_loss_mean_bps'])
            target, valid = net_labels(market, rows, 1800, side)
            valid &= rows + 1800 < bounds['stop_row']
            target[~valid] = np.nan
            profiles.append({'model': name, 'side': side, **output_profile(edge, probability, target)})
            bucket = np.digitize(edge, [-20., -10., -5., 0., 5., 10., 20.])
            for index in range(8):
                selected = valid & (bucket == index)
                if selected.any():
                    bins.append({
                        'model': name, 'side': side, 'edge_bucket': index,
                        'count': int(selected.sum()), 'predicted_edge_bps': float(edge[selected].mean()),
                        'actual_net_1800s_bps': float(target[selected].mean()),
                        'win_rate': float((target[selected] > 0).mean()),
                    })
            for lag in [1, 60, 300, 1800]:
                correlations.append({
                    'model': name, 'side': side, 'kind': 'autocorrelation', 'seconds': lag,
                    'prediction': finite_correlation(edge[:-lag], edge[lag:]),
                    'actual': finite_correlation(target[:-lag], target[lag:]),
                })
            for horizon in [60, 180, 600, 1800, 3600]:
                usable = rows + horizon < bounds['stop_row']
                future = np.full(len(rows), np.nan)
                future[usable] = side * np.log(market[rows[usable] + horizon, 0] / market[rows[usable], 0]) * 10000
                price_valid = ((market[rows, 8] > .5) & (market[rows + horizon, 8] > .5)
                               & (market[rows, 9] == market[rows + horizon, 9]))
                future[~price_valid] = np.nan
                correlations.append({
                    'model': name, 'side': side, 'kind': 'future_mid_return', 'seconds': horizon,
                    'prediction': finite_correlation(edge, future), 'actual': None,
                })
            frame = pl.DataFrame({
                'date': (timestamps // (86400 * SECOND)).astype('int64'),
                'edge': edge, 'probability': probability, 'target': target,
            }).with_columns(pl.col('target').fill_nan(None))
            daily = frame.group_by('date').agg(
                pl.len().alias('grid_rows'), pl.col('target').count().alias('label_rows'),
                pl.col('edge').mean().alias('mean_edge_bps'),
                (pl.col('edge') > 0).mean().alias('positive_edge_fraction'),
                pl.col('target').mean().alias('mean_target_bps'),
                pl.corr('edge', 'target').alias('prediction_target_correlation'),
            ).with_columns(pl.lit(name).alias('model'), pl.lit(side).alias('side'))
            daily_frames.append(daily)
            timeline = timeline.with_columns(
                pl.Series(f'edge_side{side}_bps', edge[::60]),
                pl.Series(f'probability_side{side}', probability[::60]),
                pl.Series(f'future_net_1800s_side{side}_bps', target[::60]),
            )
        direction = np.where(np.maximum(*side_predictions) > 0,
                             np.where(side_predictions[0] >= side_predictions[1], 1, -1), 0)
        changes = np.r_[0, np.flatnonzero(np.diff(direction)) + 1, len(direction)]
        durations = np.diff(changes)
        active = direction[changes[:-1]] != 0
        profiles.append({
            'model': name, 'side': 0, 'active_fraction': float((direction != 0).mean()),
            'long_fraction': float((direction > 0).mean()), 'short_fraction': float((direction < 0).mean()),
            'active_runs': int(active.sum()),
            'active_run_p50_s': float(np.median(durations[active])) if active.any() else None,
            'active_run_max_s': int(durations[active].max()) if active.any() else None,
        })
        case = Path(specification['case'])
        trades = pl.read_parquet(case.with_suffix('.trades.parquet'))
        entry_rows = np.rint(trades['entry_time_s'].to_numpy() - timestamps[0] / SECOND - .002).astype('int64')
        side = trades['side'].to_numpy()
        entry_edge = np.where(side > 0, side_predictions[0][entry_rows], side_predictions[1][entry_rows])
        trades = trades.with_columns(
            pl.lit(name).alias('model'), pl.Series('entry_edge_bps', entry_edge),
            pl.Series('entry_past_600s_vol_bps', risk[entry_rows]),
            (pl.col('exit_time_s') - pl.col('entry_time_s')).alias('holding_seconds'),
            ((pl.col('gross_pnl_usdt') > 0) & (pl.col('net_pnl_usdt') <= 0)).alias('gross_profit_lost_to_costs'),
        )
        trade_frames.append(trades)
        equity = pl.read_parquet(case.with_suffix('.timeline.parquet')).with_columns(
            ((pl.col('timestamp_s') - .002).round(0).cast(pl.Int64) * SECOND).alias('timestamp_ns'),
        )
        timeline.join(equity, on='timestamp_ns', how='left').write_parquet(output / f'{name}_timeline.parquet')
    pl.DataFrame(profiles, infer_schema_length=None).write_csv(output / 'output_profiles.csv')
    pl.DataFrame(bins).write_csv(output / 'prediction_bins.csv')
    pl.DataFrame(correlations).write_csv(output / 'temporal_correlations.csv')
    pl.concat(daily_frames).sort(['model', 'side', 'date']).write_csv(output / 'daily_predictions.csv')
    pl.concat(trade_frames).sort('net_pnl_usdt').write_parquet(output / 'trades_with_signal.parquet')
    write_json_atomic(output / 'completed.json', {
        'status': 'success', 'split': 'val_dev',
        'edge_buckets': ['<-20', '-20:-10', '-10:-5', '-5:0', '0:5', '5:10', '10:20', '>=20'],
        'fixed_label': '1800_second_bid_ask_net_5bps_each_side_without_funding',
        'trade_outcome': 'actual_holding_with_execution_fees_and_funding',
        'timeline': 'minute_samples_from_actual_ledger_not_reconstructed',
    })


# 【非核心】常数或无共同有效观测时相关未定义，报告为空而不是零相关。
def finite_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    """仅诊断使用，不据此改变训练或交易样本。"""
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return None
    first, second = left[valid], right[valid]
    if first.std() == 0 or second.std() == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


# 【核心逻辑412】把模型输出尾部和真实标签尾部并列，保留正价值区域的实际获利率。
def output_profile(edge: np.ndarray, probability: np.ndarray, target: np.ndarray) -> dict:
    """预测分布用全部网格，实际收益分布仅用有效固定期限标签。"""
    actual = target[np.isfinite(target)]
    positive = (edge > 0) & np.isfinite(target)
    result = {
        'grid_rows': len(edge), 'label_rows': len(actual),
        'mean_probability': float(probability.mean()), 'mean_edge_bps': float(edge.mean()),
        'edge_std_bps': float(edge.std()), 'actual_mean_bps': float(actual.mean()),
        'actual_std_bps': float(actual.std()), 'positive_edge_fraction': float((edge > 0).mean()),
        'positive_edge_win_rate': float((target[positive] > 0).mean()) if positive.any() else None,
    }
    for quantile in [.01, .5, .99]:
        result[f'edge_q{quantile}'] = float(np.quantile(edge, quantile))
        result[f'actual_q{quantile}'] = float(np.quantile(actual, quantile))
    return result


# 【非核心】入场状态截止上一完整秒，剔除跨质量区段的历史收益。
def entry_past_return(market: np.ndarray, entry_rows: np.ndarray, horizon: int) -> np.ndarray:
    """返回入场前horizon秒的对数bps；不读取入场所在秒或任何未来价格。"""
    last = entry_rows - 1
    first = last - horizon
    prices = market[last, 0]
    past_prices = market[first, 0]
    valid = (
        (market[last, 8] > .5)
        & (market[first, 8] > .5)
        & (market[last, 9] == market[first, 9])
        & np.isfinite(prices)
        & np.isfinite(past_prices)
        & (prices > 0)
        & (past_prices > 0)
    )
    returns = np.full(len(entry_rows), np.nan)
    returns[valid] = np.log(prices[valid] / past_prices[valid]) * 10000
    return returns


# 【核心逻辑414】按事前趋势与波动归因实际交易，不用未来收益划分行情或删除交易。
def evaluate_entry_regimes(filename: Path) -> None:
    """复用逐笔账本，固定10bps趋势分界和20bps波动分界，只诊断开发期。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    trades = pl.read_parquet(output / 'trades_with_signal.parquet')
    entry_rows = np.rint(
        trades['entry_time_s'].to_numpy() - manifest['start_timestamp_ns'] / SECOND - .002
    ).astype('int64')
    quality = pl.DataFrame([entry_window_quality(market, int(row), 1800) for row in entry_rows])
    trades = trades.hstack(quality)
    for horizon in [600, 1800]:
        past_return = entry_past_return(market, entry_rows, horizon)
        trades = trades.with_columns(pl.Series(f'past_{horizon}s_bps', past_return).fill_nan(None))
    trades = trades.with_columns(
        pl.when(pl.col('past_1800s_bps').is_null()).then(pl.lit('unknown'))
        .when(pl.col('past_1800s_bps') < -10).then(pl.lit('falling'))
        .when(pl.col('past_1800s_bps') > 10).then(pl.lit('rising'))
        .otherwise(pl.lit('flat')).alias('trend'),
        pl.when(~pl.col('entry_past_600s_vol_bps').is_finite()).then(pl.lit('unknown'))
        .when(pl.col('entry_past_600s_vol_bps') >= 20).then(pl.lit('high'))
        .otherwise(pl.lit('low')).alias('volatility'),
    )
    statistics = [
        pl.len().alias('trades'),
        pl.col('gross_pnl_usdt').sum().alias('gross_usdt'),
        pl.col('net_pnl_usdt').sum().alias('net_usdt'),
        pl.col('net_pnl_usdt').mean().alias('mean_net_usdt'),
        pl.col('net_pnl_usdt').min().alias('worst_trade_usdt'),
        (pl.col('net_pnl_usdt') > 0).mean().alias('win_rate'),
        pl.col('holding_seconds').median().alias('median_hold_s'),
    ]
    for dimension in ['trend', 'volatility']:
        trades.group_by(['model', dimension]).agg(statistics).sort(
            ['model', dimension]
        ).write_csv(output / f'trades_by_{dimension}.csv')
    trades.group_by(['model', 'trend', 'volatility']).agg(statistics).sort(
        ['model', 'trend', 'volatility']
    ).write_csv(output / 'trades_by_joint_regime.csv')
    trades.write_parquet(output / 'trades_with_regime.parquet')
    write_json_atomic(output / 'regimes_completed.json', {
        'status': 'success',
        'split': 'val_dev',
        'trend_horizon_s': 1800,
        'trend_boundary_bps': 10,
        'volatility_boundary_bps': 20,
        'purpose': 'descriptive_attribution_not_parameter_selection',
        'information_cutoff': 'previous_complete_second',
    })


# 【非核心】直接检查入场前窗口的逐秒盘口，保留区段变化及原始时钟跳跃，不推断缺失原因。
def entry_window_quality(market: np.ndarray, entry_row: int, horizon: int) -> dict:
    """输出窗口内无效秒、最长连续无效秒、区段变化数及最近变化距入场秒数。"""
    window = market[entry_row - horizon - 1:entry_row]
    invalid = window[:, 8] <= .5
    run_edges = np.flatnonzero(np.diff(np.r_[False, invalid, False]))
    invalid_lengths = run_edges[1::2] - run_edges[::2]
    changes = np.flatnonzero(np.diff(window[:, 9]) != 0) + 1
    exchange_steps = np.diff(window[:, 5]) / SECOND
    return {
        'past_invalid_seconds': int(invalid.sum()),
        'past_longest_invalid_seconds': int(invalid_lengths.max()) if len(invalid_lengths) else 0,
        'past_segment_changes': len(changes),
        'seconds_since_segment_change': int(len(window) - changes[-1]) if len(changes) else None,
        'past_max_exchange_step_s': float(np.nanmax(exchange_steps)),
        'past_max_snapshot_age_s': float(np.nanmax(window[:, 7]) / SECOND),
    }
