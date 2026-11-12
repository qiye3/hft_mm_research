"""180秒强信号的固定持有交易筛选和原生挂单成交归因。"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import DAY, HORIZONS, SECOND
from mm_research.multihorizon.inventory import funding_inputs, simulate_inventory, walk_levels
from mm_research.strategies.forecast_execution import run_probe_day
from mm_research.strategies.forecast_research import research_context
from mm_research.strategies.tradeability import dataset_definition


# 【非核心】将快照逐笔成交关联成完整往返，资金费采用收取时刻实际持仓。
def snapshot_round_trips(fills: np.ndarray, funding: tuple) -> pl.DataFrame:
    """部分平仓沿用同一交易编号，空仓后的下一次开仓才生成新编号。"""
    schema = {
        'trade_id': pl.Int64, 'entry_time_s': pl.Float64, 'exit_time_s': pl.Float64,
        'side': pl.Float64, 'quantity_btc': pl.Float64, 'entry_notional_usdt': pl.Float64,
        'gross_pnl_usdt': pl.Float64, 'fees_usdt': pl.Float64, 'funding_paid_usdt': pl.Float64,
        'net_pnl_usdt': pl.Float64, 'net_return_bps': pl.Float64,
    }
    if len(fills) == 0:
        return pl.DataFrame(schema=schema)
    identifiers = np.cumsum(np.r_[True, np.abs(fills[:-1, 5]) < 1e-10]).astype('int64')
    times, rates, marks = funding
    funding_per_trade = np.zeros(identifiers[-1] + 1)
    for timestamp, rate, mark in zip(times, rates, marks):
        previous = np.searchsorted(fills[:, 0], timestamp, side='left') - 1
        if previous >= 0:
            funding_per_trade[identifiers[previous]] += fills[previous, 5] * rate * mark
    frame = pl.DataFrame({
        'trade_id': identifiers, 'time_s': fills[:, 0], 'side': fills[:, 1],
        'quantity': fills[:, 2], 'price': fills[:, 3], 'fee': fills[:, 4],
    }).with_columns(
        (-pl.col('side') * pl.col('quantity') * pl.col('price')).alias('cashflow'),
    )
    trades = frame.group_by('trade_id', maintain_order=True).agg(
        pl.col('time_s').first().alias('entry_time_s'),
        pl.col('time_s').last().alias('exit_time_s'),
        pl.col('side').first(), pl.col('quantity').first().alias('quantity_btc'),
        (pl.col('quantity') * pl.col('price')).first().alias('entry_notional_usdt'),
        pl.col('cashflow').sum().alias('gross_pnl_usdt'), pl.col('fee').sum().alias('fees_usdt'),
    ).with_columns(
        pl.Series('funding_paid_usdt', funding_per_trade[1:]),
    ).with_columns(
        (pl.col('gross_pnl_usdt') - pl.col('fees_usdt') - pl.col('funding_paid_usdt')).alias('net_pnl_usdt'),
    ).with_columns(
        (pl.col('net_pnl_usdt') / pl.col('entry_notional_usdt') * 10000).alias('net_return_bps'),
    )
    return trades.select(list(schema))


# 【核心逻辑338】复用已有逐档快照模拟，只在强信号触发时开仓并固定持有180秒。
def screen_signal_trades(filename: Path, signal_id: str, split: str) -> None:
    """同一连续分区保留跨日持仓，空仓才开仓，实际成交及资金费共同决定净收益。"""
    config, source, output = research_context(filename)
    manifest, segments = dataset_definition(source)
    bounds = segments[split]
    thresholds = json.loads((output / 'thresholds.json').read_text())['signals'][signal_id]['quantiles']
    frame = pl.read_parquet(output / 'signals' / f'{signal_id}_{split}.parquet')
    books = np.load(source / 'e3' / f'{split}_books.npy', mmap_mode='r')
    if len(books) != frame.height:
        raise ValueError('预测和到达时刻盘口行数不同')
    funding = funding_inputs(source, manifest, bounds['start_row'], bounds['stop_row'])
    settings = config['execution']
    latency = config['prediction_latency_ns'] + settings['entry_latency_ns']
    times = frame['timestamp_ns'].to_numpy().astype('float64') / SECOND + latency / SECOND
    horizon_index = int(np.flatnonzero(HORIZONS == config['hold_s'])[0])
    folder = output / 'screening'
    folder.mkdir(exist_ok=True)
    for quantile in config['quantiles']:
        identifier = f'{signal_id}_q{int(quantile * 100)}_{split}'
        marker = folder / f'{identifier}.json'
        if marker.exists():
            continue
        alpha = frame['prediction_bps'].to_numpy()
        eligible = np.isfinite(alpha) & (np.abs(alpha) >= thresholds[str(quantile)])
        eligible[:120] = False
        eligible[-config['hold_s'] - 20:] = False
        predictions = np.zeros((frame.height, len(HORIZONS)))
        predictions[:, horizon_index] = np.where(eligible, np.sign(alpha), 0.)
        # P0按给定方向固定持有，零方向使空仓保持空仓，不把未来标签用于决策。
        ledger, fills, final_position = simulate_inventory(
            predictions, books, times, np.ones(len(HORIZONS)), 0, horizon_index,
            config['hold_s'], 1., 0., 1., settings['risk_unit_btc'], settings['taker_fee'],
            settings['minimum_notional_usdt'], settings['quantity_step_btc'], config['hold_s'],
            settings['initial_equity_usdt'], *funding,
        )
        if abs(final_position) >= settings['quantity_step_btc'] * .5:
            raise ValueError(f'固定持有筛选未完成平仓 {identifier}')
        gross = float(-(fills[:, 1] * fills[:, 2] * fills[:, 3]).sum())
        fees = float(fills[:, 4].sum())
        paid_funding = float(ledger[-1, 6])
        equity = ledger[:, 7]
        ending = float(equity[np.isfinite(equity)][-1])
        if abs(gross - fees - paid_funding - ending) > 1e-6:
            raise ValueError(f'快照筛选账本不一致 {identifier}')
        trades = snapshot_round_trips(fills, funding)
        if abs(trades['net_pnl_usdt'].sum() - ending) > 1e-6:
            raise ValueError(f'往返交易损益不一致 {identifier}')
        trades.write_parquet(folder / f'{identifier}.trades.parquet', compression='zstd')
        daily = []
        previous_equity = 0.
        for begin in range(0, len(ledger), DAY):
            stop = min(begin + DAY, len(ledger))
            valid_equity = ledger[begin:stop, 7]
            valid_equity = valid_equity[np.isfinite(valid_equity)]
            if len(valid_equity) == 0:
                raise ValueError(f'整日无可执行估值 {identifier} {begin}')
            daily_equity = float(valid_equity[-1])
            daily.append({
                'date': manifest['dates'][(bounds['start_row'] + begin) // DAY],
                'net_pnl_usdt': daily_equity - previous_equity,
                'ending_liquidation_pnl_usdt': daily_equity,
                'ending_position_btc': float(ledger[stop - 1, 3]),
            })
            previous_equity = daily_equity
        pl.DataFrame(daily).write_csv(folder / f'{identifier}.daily.csv')
        finite_equity = equity[np.isfinite(equity)]
        drawdown = np.maximum.accumulate(np.r_[0., finite_equity])[1:] - finite_equity
        write_json_atomic(marker, {
            'id': identifier, 'signal': signal_id, 'quantile': quantile, 'split': split,
            'execution_type': 'arrival_time_backward_snapshot_depth_walk',
            'account_reset': 'segment_start_only',
            'round_trips': trades.height, 'gross_pnl_usdt': gross, 'fees_usdt': fees,
            'funding_paid_usdt': paid_funding, 'net_pnl_usdt': ending,
            'zero_fee_same_fills_pnl_usdt': gross - paid_funding,
            'max_drawdown_usdt': float(drawdown.max()),
            'max_abs_position_btc': float(np.abs(ledger[:, 3]).max()),
            'final_position_btc': final_position,
        })
        print('SCREENING', identifier, ending, trades.height, flush=True)


# 【核心逻辑339】按完整开发区间净收益保留至多两个候选，没有正收益时转入固定机制诊断。
def select_trade_candidates(filename: Path) -> None:
    """筛选不读取复核期收益，所有九组原始结果保留供后续对照。"""
    config, source, output = research_context(filename)
    records = [json.loads(path.read_text()) for path in (output / 'screening').glob('*_val_dev.json')]
    scores = pl.DataFrame(records).sort(['net_pnl_usdt', 'id'], descending=[True, False])
    scores.write_csv(output / 'development_screening.csv')
    positive = scores.filter((pl.col('net_pnl_usdt') > 0) & (pl.col('round_trips') > 0))
    selected = positive.head(config['maximum_candidates']).select('signal', 'quantile').to_dicts()
    write_json_atomic(output / 'selected_candidates.json', {
        'selected_on': 'val_dev_only', 'candidates': selected,
        'next_stage': 'candidate_execution' if selected else 'fixed_mechanism_diagnostic',
        'mechanism_signal': 'h180_old_mlp',
    })


# 【核心逻辑340】原生框架固定提交时刻和方向，独立记录未成交、部分成交与真实180秒退出。
def run_mechanism_day(filename: Path, date: str) -> None:
    """复用既有固定订单程序与已修复引擎，三个挂价和等待组合不由当日收益选择。"""
    config, source, output = research_context(filename)
    for name, target in [('l2', source / 'l2'), ('build', Path(config['forecast_source']) / 'build')]:
        destination = output / name
        if not destination.exists():
            destination.symlink_to(target.resolve(), target_is_directory=True)
    marker = output / 'probes' / f'{date}.completed.json'
    if not marker.exists():
        run_probe_day(filename, date)


# 【核心逻辑343】按实际决策可见信号解释固定订单价值，门槛只用训练期定义。
def summarize_mechanisms(filename: Path) -> None:
    """全部机会与成交条件收益分别保存，未成交的实际净收益保持为零。"""
    config, source, output = research_context(filename)
    manifest, segments = dataset_definition(source)
    thresholds = json.loads((output / 'thresholds.json').read_text())['signals']['h180_old_mlp']['quantiles']
    labels = np.load(source / 'e0/y.npy', mmap_mode='r')
    records = []
    for split, dates in config['mechanism_dates'].items():
        for date in dates:
            daily_records = []
            alpha = pl.read_csv(Path(config['forecast_source']) / 'signals' / f'h180_old_mlp_{date}.csv')
            alpha = alpha.select(
                (pl.col('timestamp_ns') + config['prediction_latency_ns']).alias('ready_ns'),
                pl.col('prediction_bps').alias('alpha_bps'),
            )
            for path in sorted((output / 'probes').glob(f'*{date}*.parquet')):
                table = pl.read_parquet(path).sort('decision_ts_ns').join_asof(
                    alpha.sort('ready_ns'), left_on='decision_ts_ns', right_on='ready_ns',
                    strategy='backward', tolerance=SECOND,
                )
                rows = ((table['decision_ts_ns'].to_numpy() - manifest['start_timestamp_ns']) // SECOND).astype('int64')
                index = int(np.flatnonzero(HORIZONS == 180)[0])
                table = table.with_columns(
                    pl.Series('future_180s_signed_bps', labels[rows, index] * table['side'].to_numpy()),
                    (pl.col('side') * pl.col('alpha_bps')).alias('signed_alpha_bps'),
                    pl.lit(split).alias('split'),
                )
                daily_records.append(table)
            daily = pl.concat(daily_records, how='diagonal_relaxed')
            active = active_opportunity_reference(config, source, daily, date)
            records.append(daily.join(active, on='candidate_id', how='left', validate='1:1'))
    opportunities = pl.concat(records, how='diagonal_relaxed')
    opportunities.write_parquet(output / 'mechanism_opportunities.parquet', compression='zstd')
    groups = []
    combined_groups = []
    future = pl.col('future_180s_signed_bps')
    common = pl.col('active_status') == 'complete'
    aggregations = [
        pl.len().alias('opportunities'), pl.col('submitted').sum().alias('submitted_opportunities'),
        pl.col('filled_any').sum().alias('filled_opportunities'),
        pl.col('filled_any').mean().alias('fill_probability'), pl.col('fill_fraction').mean(),
        future.filter(future.is_finite()).mean().alias('all_future_180s_bps'),
        future.filter(pl.col('filled_any') & future.is_finite()).mean().alias('filled_future_180s_bps'),
        pl.col('markout_180s_bps').mean(), pl.col('submitted_value_bps').mean(),
        pl.col('conditional_net_bps').mean(), pl.col('active_net_return_bps').mean(),
        pl.col('submitted_value_bps').filter(common).mean().alias('maker_value_on_common_opportunities_bps'),
        pl.col('active_net_return_bps').filter(pl.col('filled_any')).mean().alias('active_return_on_maker_filled_bps'),
        pl.col('active_net_return_bps').filter(~pl.col('filled_any')).mean().alias('active_return_on_maker_unfilled_bps'),
        common.sum().alias('active_complete_opportunities'), pl.col('net_pnl_usdt').sum(),
    ]
    for name, threshold in [('same_direction', 0.), *[(f'q{int(float(key) * 100)}', value) for key, value in thresholds.items()]]:
        selected = opportunities.filter(pl.col('cycle_complete') & (pl.col('signed_alpha_bps') >= threshold))
        summary = selected.group_by('split', 'date', 'offset_ticks', 'ttl_ms').agg(aggregations).with_columns(
            pl.lit(name).alias('signal_group'),
        )
        groups.append(summary)
        combined = selected.group_by('split', 'offset_ticks', 'ttl_ms').agg(aggregations).with_columns(
            pl.lit(name).alias('signal_group'),
        )
        combined_groups.append(combined)
    pl.concat(groups).write_csv(output / 'mechanism_daily.csv')
    pl.concat(combined_groups).sort('split', 'signal_group', 'offset_ticks', 'ttl_ms').write_csv(output / 'mechanism_summary.csv')


# 【核心逻辑344】为相同提交机会补算主动进出对照，严格使用到达时刻之前的真实十档盘口。
def active_opportunity_reference(config: dict, source: Path, opportunities: pl.DataFrame, date: str) -> pl.DataFrame:
    """以实际发单时刻为锚点，开仓到达后持有180秒，缺少报价或退出深度时明确标记。"""
    manifest, segments = dataset_definition(source)
    columns = [f'{side}_{field}_{level}' for side in ['bid', 'ask']
               for field in ['price', 'quantity'] for level in range(1, 11)]
    snapshots = pl.scan_parquet(Path(config['standard_folder']) / 'snapshots' / f'{date}.parquet').select(
        'exchange_timestamp_ns', *columns,
    ).collect(engine='streaming').sort('exchange_timestamp_ns').unique(
        'exchange_timestamp_ns', keep='last', maintain_order=True,
    )
    entries = opportunities.select('candidate_id', 'decision_ts_ns', 'side', 'submitted_qty_btc')
    entries = entries.with_columns(
        (pl.col('decision_ts_ns') + config['execution']['entry_latency_ns']).alias('entry_arrival_ns'),
    ).with_columns(
        (pl.col('entry_arrival_ns') + config['hold_s'] * SECOND).alias('exit_arrival_ns'),
    )
    entry_books = entries.sort('entry_arrival_ns').join_asof(
        snapshots, left_on='entry_arrival_ns', right_on='exchange_timestamp_ns',
        strategy='backward', tolerance=SECOND,
    ).select('candidate_id', *columns)
    exit_books = entries.sort('exit_arrival_ns').join_asof(
        snapshots, left_on='exit_arrival_ns', right_on='exchange_timestamp_ns',
        strategy='backward', tolerance=SECOND,
    ).select('candidate_id', *columns)
    entry_map = dict(zip(entry_books['candidate_id'].to_list(), entry_books.select(columns).to_numpy()))
    exit_map = dict(zip(exit_books['candidate_id'].to_list(), exit_books.select(columns).to_numpy()))
    day = manifest['dates'].index(date)
    funding_times, funding_rates, funding_marks = funding_inputs(source, manifest, day * DAY, (day + 1) * DAY)
    results = []
    for opportunity in entries.iter_rows(named=True):
        identifier = opportunity['candidate_id']
        entry_book = entry_map[identifier]
        exit_book = exit_map[identifier]
        record = {'candidate_id': identifier, 'active_status': 'missing_snapshot', 'active_net_return_bps': None}
        if np.isfinite(entry_book).all() and np.isfinite(exit_book).all():
            direction = opportunity['side']
            entry_offset = 20 if direction > 0 else 0
            exit_offset = 0 if direction > 0 else 20
            quantity, entry_price = walk_levels(
                entry_book[entry_offset:entry_offset + 10], entry_book[entry_offset + 10:entry_offset + 20],
                opportunity['submitted_qty_btc'], config['execution']['quantity_step_btc'],
            )
            closed, exit_price = walk_levels(
                exit_book[exit_offset:exit_offset + 10], exit_book[exit_offset + 10:exit_offset + 20],
                quantity, config['execution']['quantity_step_btc'],
            )
            record['active_status'] = 'insufficient_depth'
            if quantity > 0 and abs(closed - quantity) < 1e-10:
                entry_seconds = opportunity['entry_arrival_ns'] / SECOND
                exit_seconds = opportunity['exit_arrival_ns'] / SECOND
                funding_mask = (funding_times > entry_seconds) & (funding_times <= exit_seconds)
                paid = float((funding_rates[funding_mask] * funding_marks[funding_mask]).sum() * direction * quantity)
                gross = direction * quantity * (exit_price - entry_price)
                fees = quantity * (entry_price + exit_price) * config['execution']['taker_fee']
                record['active_status'] = 'complete'
                record['active_net_return_bps'] = (gross - fees - paid) / (quantity * entry_price) * 10000
        results.append(record)
    return pl.DataFrame(results, schema={
        'candidate_id': pl.String, 'active_status': pl.String, 'active_net_return_bps': pl.Float64,
    })


# 【非核心】研究入口按明确阶段转交对应计算，日期与模型由后台任务配置给出。
def run_tradeability_stage(filename: Path, stage: str, signal: str | None, split: str | None, date: str | None) -> None:
    """报告在上游真实终态后生成，队列错误保留为错误。"""
    if stage == 'screen':
        if signal is None or split not in ['val_dev', 'val_select']:
            raise ValueError('screen需要明确的signal和开发或复核分区')
        screen_signal_trades(filename, signal, split)
    elif stage == 'select':
        select_trade_candidates(filename)
    elif stage == 'mechanism':
        if date is None:
            raise ValueError('mechanism需要明确日期')
        run_mechanism_day(filename, date)
    elif stage == 'report':
        from mm_research.strategies.tradeability_report import write_tradeability_report
        write_tradeability_report(filename)
