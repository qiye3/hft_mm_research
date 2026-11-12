"""180秒信号的训练期尾部门槛、逐日条件收益和成本可达性。"""

import json
from pathlib import Path

from numba import njit
import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import DAY, HORIZONS, SECOND
from mm_research.strategies.forecast_research import research_context


# 【非核心】从同一冻结矩阵读取时间定义，所有研究阶段共用原始时间分区。
def dataset_definition(source: Path) -> tuple[dict, dict]:
    """返回数据清单与分区，不复制市场矩阵或标签。"""
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    segments = json.loads((source / 'e0/splits.json').read_text())['segments']
    return manifest, segments


# 【核心逻辑330】只用训练前向预测固定极值门槛和市场状态边界，沿用已完成的幅度校准。
def freeze_tail_thresholds(filename: Path) -> None:
    """输出每个信号的q90/q95/q99，状态中位数来自训练期固定60秒网格。"""
    config, source, output = research_context(filename)
    if (output / 'thresholds.json').exists():
        return
    manifest, segments = dataset_definition(source)
    forecast = Path(config['forecast_source'])
    calibrations = json.loads((forecast / 'calibration.json').read_text())
    folds = json.loads((source / 'e0/splits.json').read_text())['oof_folds']
    thresholds = {}
    for signal in config['signals']:
        samples = []
        calibration = calibrations[signal['id']]
        for fold in folds:
            frame = pl.read_parquet(forecast / 'oof' / signal['id'] / fold['id'] / 'predictions.parquet')
            first = manifest['start_timestamp_ns'] + fold['predict_start'] * SECOND
            last = manifest['start_timestamp_ns'] + fold['predict_stop'] * SECOND
            values = frame.filter(
                pl.col('timestamp_ns').is_between(first, last, closed='left')
                & pl.col('target_log_bps').is_finite()
                & pl.col('prediction_log_bps').is_finite(),
            ).select(
                (pl.col('prediction_log_bps') * calibration['slope'] + calibration['intercept']).abs(),
            ).to_series().to_numpy()
            samples.append(values)
        values = np.concatenate(samples)
        thresholds[signal['id']] = {
            'quantiles': dict(zip(map(str, config['quantiles']), np.quantile(values, config['quantiles']).tolist())),
            'calibration': calibration,
            'oof_samples': len(values),
        }
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    stop = segments['train']['stop_row'] - 300
    states = market_states(market, manifest, 0, stop, config)
    sample = states.slice(0, stop).gather_every(config['state_training_stride_s'])
    state_boundaries = sample.filter(pl.col('book_valid')).select(
        pl.col('volatility_60s_bps').median().alias('volatility_median_bps'),
        pl.col('spread_bps').median().alias('spread_median_bps'),
    ).row(0, named=True)
    (output / 'signals').mkdir(exist_ok=True)
    for signal in config['signals']:
        for split in ['val_dev', 'val_select']:
            destination = output / 'signals' / f'{signal["id"]}_{split}.parquet'
            if not destination.exists():
                destination.symlink_to((forecast / 'signals' / destination.name).resolve())
    write_json_atomic(output / 'thresholds.json', {
        'fit': 'train_forward_oof',
        'signals': thresholds,
        'states': state_boundaries,
        'thresholds_fixed_after_utc': segments['train']['end_utc_exclusive'],
    })


# 【核心逻辑331】使用采样时刻已可见市场状态，波动窗口跨午夜延续并以缺失值控制有效数。
def market_states(market: np.ndarray, manifest: dict, begin: int, stop: int, config: dict) -> pl.DataFrame:
    """返回当前价差和向后60秒收益波动，前置历史只用于滚动预热。"""
    prefix = max(0, begin - config['state_lookback_s'])
    columns = {name: index for index, name in enumerate(manifest['market_columns'])}
    mid = market[prefix:stop, columns['mid_price']]
    bid = market[prefix:stop, columns['bid_price_1']]
    ask = market[prefix:stop, columns['ask_price_1']]
    valid = market[prefix:stop, columns['book_valid']] == 1
    valid &= np.isfinite(mid) & (mid > 0) & (bid < ask)
    frame = pl.DataFrame({
        'mid': mid,
        'spread_bps': np.where(valid, (ask - bid) / mid * 10000, np.nan),
        'book_valid': valid,
    }).with_columns(
        pl.when(pl.col('book_valid')).then(pl.col('mid').log()).otherwise(None).alias('log_mid'),
    ).with_columns(
        pl.col('log_mid').diff().rolling_std(
            config['state_lookback_s'], min_samples=config['state_min_samples'],
        ).mul(10000).alias('volatility_60s_bps'),
    )
    return frame.slice(begin - prefix).select('book_valid', 'spread_bps', 'volatility_60s_bps')


# 【核心逻辑335】强信号首次触发后等待固定持有期，独立机会的选择不读取未来收益。
@njit(cache=True)
def independent_opportunities(timestamps: np.ndarray, selected: np.ndarray, hold_ns: int) -> np.ndarray:
    """返回不重叠机会掩码，机会从信号时刻开始计时，真实成交持有期在交易阶段计算。"""
    opportunities = np.zeros(len(timestamps), dtype=np.bool_)
    available_at = np.int64(-9223372036854775807)
    for row in range(len(timestamps)):
        if selected[row] and timestamps[row] >= available_at:
            opportunities[row] = True
            available_at = timestamps[row] + hold_ns
    return opportunities


# 【非核心】按日期保存足够的收益统计量，Bootstrap以日期为单位保留日内依赖。
def daily_tail_statistics(frame: pl.DataFrame, horizons: list[int]) -> pl.DataFrame:
    """同时保留全部强信号行与不重叠机会，未来标签只影响对应期限的评价。"""
    aggregations = [
        pl.len().alias('signal_rows'),
        pl.col('opportunity').sum().alias('opportunities'),
        pl.col('prediction_bps').abs().mean().alias('mean_abs_prediction_bps'),
        pl.col('prediction_bps').abs().max().alias('max_abs_prediction_bps'),
        pl.col('spread_bps').mean().alias('mean_spread_bps'),
    ]
    for horizon in horizons:
        value = pl.col(f'y_{horizon}s_bps') * pl.col('side')
        valid = pl.col(f'y_{horizon}s_bps').is_finite() & pl.col('evaluation_valid')
        opportunity = valid & pl.col('opportunity')
        aggregations.extend([
            valid.sum().alias(f'n_{horizon}s'),
            value.filter(valid).sum().alias(f'sum_{horizon}s_bps'),
            opportunity.sum().alias(f'opportunity_n_{horizon}s'),
            value.filter(opportunity).sum().alias(f'opportunity_sum_{horizon}s_bps'),
            (value.filter(opportunity) > 0).sum().alias(f'opportunity_wins_{horizon}s'),
        ])
    return frame.group_by('date').agg(aggregations)


# 【核心逻辑334】逐月诊断三个固定强度与单独市场状态，所有选择沿用训练期边界。
def analyze_signal_tails(filename: Path, signal_id: str, split: str) -> None:
    """输出逐日条件统计，不在当月重新定义分位门槛，不按未来标签选择发单机会。"""
    config, source, output = research_context(filename)
    folder = output / 'tails'
    folder.mkdir(exist_ok=True)
    destination = folder / f'{signal_id}_{split}.parquet'
    if destination.exists():
        return
    manifest, segments = dataset_definition(source)
    thresholds = json.loads((output / 'thresholds.json').read_text())
    boundaries = thresholds['states']
    predictions = pl.read_parquet(output / 'signals' / f'{signal_id}_{split}.parquet')
    bounds = segments[split]
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    labels = np.load(source / 'e0/y.npy', mmap_mode='r')
    records = []
    available_at = {quantile: 0 for quantile in config['quantiles']}
    for month in sorted({date[:7] for date in manifest['dates'][bounds['start_row'] // DAY:bounds['stop_row'] // DAY]}):
        frame = predictions.filter(pl.from_epoch('timestamp_ns', time_unit='ns').dt.strftime('%Y-%m') == month)
        first = int((frame['timestamp_ns'][0] - manifest['start_timestamp_ns']) // SECOND)
        stop = first + frame.height
        states = market_states(market, manifest, first, stop, config)
        frame = frame.hstack(states).with_columns(
            pl.from_epoch('timestamp_ns', time_unit='ns').dt.strftime('%Y-%m-%d').alias('date'),
            pl.col('prediction_bps').sign().cast(pl.Int8).alias('side'),
        )
        for horizon in config['horizons_s']:
            index = int(np.flatnonzero(HORIZONS == horizon)[0])
            frame = frame.with_columns(pl.Series(f'y_{horizon}s_bps', labels[first:stop, index]))
        states_to_check = {
            'all': pl.lit(True),
            'low_volatility': pl.col('volatility_60s_bps') <= boundaries['volatility_median_bps'],
            'high_volatility': pl.col('volatility_60s_bps') > boundaries['volatility_median_bps'],
            'narrow_spread': pl.col('spread_bps') <= boundaries['spread_median_bps'],
            'wide_spread': pl.col('spread_bps') > boundaries['spread_median_bps'],
        }
        for quantile in config['quantiles']:
            threshold = thresholds['signals'][signal_id]['quantiles'][str(quantile)]
            selected = (frame['prediction_bps'].abs() >= threshold) & frame['book_valid']
            eligible = selected & (frame['timestamp_ns'] >= available_at[quantile])
            opportunities = independent_opportunities(
                frame['timestamp_ns'].to_numpy(), eligible.to_numpy(), config['hold_s'] * SECOND,
            )
            if opportunities.any():
                available_at[quantile] = int(frame['timestamp_ns'].to_numpy()[opportunities][-1]) + config['hold_s'] * SECOND
            strong = frame.with_columns(pl.Series('opportunity', opportunities)).filter(selected)
            for state, condition in states_to_check.items():
                subset = strong.filter(condition)
                for direction in [0, 1, -1]:
                    directional = subset if direction == 0 else subset.filter(pl.col('side') == direction)
                    if directional.is_empty():
                        continue
                    daily = daily_tail_statistics(directional, config['horizons_s']).with_columns(
                        pl.lit(signal_id).alias('signal'), pl.lit(split).alias('split'),
                        pl.lit(quantile).alias('quantile'), pl.lit(state).alias('state'),
                        pl.lit(direction).alias('direction'), pl.lit(month).alias('month'),
                    )
                    records.append(daily)
        print('TAIL_MONTH', signal_id, split, month, flush=True)
    temporary = folder / f'{signal_id}_{split}.partial.parquet'
    pl.concat(records).write_parquet(temporary, compression='zstd')
    temporary.replace(destination)


# 【核心逻辑337】对日内重叠收益按日期分块重采样，权重仍以机会数计算。
def daily_bootstrap(counts: np.ndarray, totals: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    """返回机会加权平均收益的95%区间，空机会的重采样不转为零收益。"""
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(counts), size=(repeats, len(counts)))
    sampled_counts = counts[indices].sum(axis=1)
    sampled_totals = totals[indices].sum(axis=1)
    averages = sampled_totals[sampled_counts > 0] / sampled_counts[sampled_counts > 0]
    return tuple(np.quantile(averages, [.025, .975]).tolist())


# 【核心逻辑336】只按开发区间判断主动交易成本可达性，复核期不反向影响候选选择。
def summarize_tails(filename: Path) -> None:
    """所有月份保留，成本门槛采用双边Taker费用，价差和滑点在交易筛选阶段另计。"""
    config, source, output = research_context(filename)
    files = sorted(path for path in (output / 'tails').glob('*.parquet') if '.partial.' not in path.name)
    daily = pl.read_parquet(files)
    manifest, segments = dataset_definition(source)
    records = []
    keys = ['signal', 'split', 'quantile', 'state', 'direction']
    for group, frame in daily.group_by(keys):
        bounds = segments[group[1]]
        calendar = pl.DataFrame({'date': manifest['dates'][bounds['start_row'] // DAY:bounds['stop_row'] // DAY]})
        active_days = frame['date'].n_unique()
        frame = calendar.join(frame, on='date', how='left')
        for horizon in config['horizons_s']:
            counts = frame[f'opportunity_n_{horizon}s'].fill_null(0).to_numpy()
            totals = frame[f'opportunity_sum_{horizon}s_bps'].fill_null(0).to_numpy()
            if counts.sum() == 0:
                continue
            lower, upper = daily_bootstrap(counts, totals, config['bootstrap_repeats'], config['seed'])
            count = int(frame[f'n_{horizon}s'].sum())
            records.append({
                **dict(zip(keys, group)),
                'horizon_s': horizon,
                'days': frame.height,
                'active_days': active_days,
                'signal_rows': int(frame['signal_rows'].sum()),
                'evaluation_rows': count,
                'trigger_share_all_grid': float(frame['signal_rows'].sum() / (bounds['stop_row'] - bounds['start_row'])),
                'signal_mean_signed_bps': frame[f'sum_{horizon}s_bps'].sum() / count if count else None,
                'opportunities': int(counts.sum()),
                'opportunity_mean_signed_bps': float(totals.sum() / counts.sum()),
                'opportunity_ci95_low_bps': lower,
                'opportunity_ci95_high_bps': upper,
            })
    summary = pl.DataFrame(records).sort(keys + ['horizon_s'])
    summary.write_csv(output / 'tail_summary.csv')
    fee_bps = 2 * config['execution']['taker_fee'] * 10000
    development = summary.filter(
        (pl.col('split') == 'val_dev') & (pl.col('state') == 'all')
        & (pl.col('direction') == 0) & (pl.col('horizon_s') == config['hold_s']),
    ).with_columns(
        (pl.col('opportunity_mean_signed_bps') > fee_bps).alias('mean_exceeds_taker_fees'),
        (pl.col('opportunity_ci95_high_bps') < fee_bps).alias('ci_below_taker_fees'),
    )
    development.write_csv(output / 'cost_feasibility.csv')
    write_json_atomic(output / 'tail_gate.json', {
        'selected_on': 'val_dev_only',
        'taker_roundtrip_fee_bps': fee_bps,
        'active_cost_reachable': development.filter(pl.col('mean_exceeds_taker_fees')).height > 0,
        'all_candidates_ci_below_fees': development['ci_below_taker_fees'].all(),
        'historical_test_status': 'previously_inspected_historical_reassessment',
    })
