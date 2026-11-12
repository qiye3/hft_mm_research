"""按入场前波动环境和持仓路径解释费用分类策略的收益来源。"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import context, SECOND


# 【非核心】仅读取指定时刻以前的完整600秒，按Train共同质量口径计算实现波动。
def past_volatility(market: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """返回600秒对数收益平方和开方的bps值，缺失或跨质量边界时保留NaN。"""
    volatility = np.full(len(rows), np.nan)
    offsets = np.arange(-601, 0)
    for begin in range(0, len(rows), 4096):
        stop = min(begin + 4096, len(rows))
        positions = rows[begin:stop, None] + offsets
        prices = market[positions, 0]
        valid = np.all(np.isfinite(prices) & (prices > 0) & (market[positions, 8] > .5), axis=1)
        segments = market[positions, 9]
        valid &= np.all(segments == segments[:, :1], axis=1)
        returns = np.diff(np.log(np.where(prices > 0, prices, np.nan)), axis=1)
        values = np.sqrt(np.sum(returns ** 2, axis=1)) * 10000
        volatility[begin:stop] = np.where(valid, values, np.nan)
    return volatility


# 【核心逻辑378】冻结Train波动分位后，对完整开发和复核交易作收益来源及尾部归因。
def attribute_cost_trades(filename: Path) -> None:
    """读取既有快照交易账本和秒级盘口，输出逐笔诊断与分组收益，不改变信号或交易规则。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    segments = json.loads((source / 'e0/splits.json').read_text())['segments']
    origin = manifest['start_timestamp_ns'] / SECOND
    train = segments['train']
    train_rows = np.arange(max(train['start_row'] + 601, 601), train['stop_row'], 60)
    train_volatility = past_volatility(market, train_rows)
    thresholds = np.nanquantile(train_volatility, [1 / 3, 2 / 3])
    write_json_atomic(output / 'volatility_definition.json', {
        'window_seconds': 600, 'train_sample_seconds': 60,
        'thresholds_bps': thresholds.tolist(), 'sample_timing': 'strictly_before_entry_second',
        'path_marking': 'one_second_opposite_best_quote_before_funding',
        'purpose': 'diagnosis_only_no_rule_selection',
    })
    rows = []
    model_config = json.loads((Path(config['profit_output']) / 'config.json').read_text())
    for split in ['val_dev', 'val_select']:
        path = Path(config['profit_output']) / 'cases' / f'{config["case"]}_{split}.trades.parquet'
        trades = pl.read_parquet(path)
        signals = pl.read_parquet(Path(model_config['models']) / f'signals/cost_h1800_{split}.parquet')
        entry_rows = np.floor(trades['entry_time_s'].to_numpy() - origin).astype('int64')
        volatility = past_volatility(market, entry_rows)
        for index, trade in enumerate(trades.iter_rows(named=True)):
            entry_price = trade['entry_notional_usdt'] / trade['quantity_btc']
            first = entry_rows[index] + 1
            stop = int(np.floor(trade['exit_time_s'] - origin)) + 1
            quote_column = 1 if trade['side'] > 0 else 2
            quotes = market[first:stop, quote_column]
            valid = np.isfinite(quotes) & (quotes > 0) & (market[first:stop, 8] > .5)
            ratio = quotes[valid] / entry_price
            marked_return = trade['side'] * (ratio - 1) * 10000 - 5 * (1 + ratio)
            regime = 'missing'
            if np.isfinite(volatility[index]):
                regime = ['low', 'medium', 'high'][int(np.searchsorted(thresholds, volatility[index]))]
            signal_row = entry_rows[index] - segments[split]['start_row']
            signal = float(signals['expected_net_edge_bps'][int(signal_row)])
            record = dict(trade)
            record.update(
                split=split, volatility_regime=regime, past_600s_volatility_bps=float(volatility[index]),
                entry_signal_bps=signal,
                worst_quote_return_bps=float(np.min(marked_return)) if len(marked_return) else None,
                best_quote_return_bps=float(np.max(marked_return)) if len(marked_return) else None,
            )
            rows.append(record)
    table = classify_failures(pl.DataFrame(rows))
    table.write_parquet(output / 'trades.parquet', compression='zstd')
    table.group_by(['split', 'side', 'volatility_regime']).agg(
        pl.len().alias('round_trips'),
        pl.col('net_pnl_usdt').sum(),
        pl.col('net_pnl_usdt').min().alias('worst_trade_usdt'),
        pl.col('worst_quote_return_bps').min(),
        pl.col('net_return_bps').mean(),
    ).sort(['split', 'side', 'volatility_regime']).write_csv(output / 'groups.csv')
    write_diagnostics(table, config, output)
    write_json_atomic(output / 'completed.json', {'status': 'success', 'trades': table.height})


# 【非核心】区分方向亏损、成本吞噬和盈利回吐，保留可重叠的路径风险标记。
def classify_failures(table: pl.DataFrame) -> pl.DataFrame:
    """输入逐笔净损益及持仓路径，输出互斥结果类别和独立风险标记。"""
    return table.with_columns(
        pl.from_epoch(pl.col('entry_time_s').cast(pl.Int64), time_unit='s').dt.strftime('%Y-%m').alias('month'),
        pl.when(pl.col('net_pnl_usdt') > 0).then(pl.lit('winner'))
        .when(pl.col('gross_pnl_usdt') > 0).then(pl.lit('cost_loss'))
        .otherwise(pl.lit('direction_loss')).alias('outcome'),
        ((pl.col('net_pnl_usdt') <= 0) & (pl.col('best_quote_return_bps') > 0)).alias('profit_giveback'),
        ((pl.col('net_pnl_usdt') > 0) & (pl.col('worst_quote_return_bps') < -50)).alias('recovered_after_50bps_adverse'),
        pl.when(pl.col('entry_signal_bps').abs() < 5).then(pl.lit('0_to_5bps'))
        .when(pl.col('entry_signal_bps').abs() < 10).then(pl.lit('5_to_10bps'))
        .otherwise(pl.lit('at_least_10bps')).alias('signal_strength'),
    )


# 【核心逻辑379】按方向、月份、信号强弱和失败原因汇总全部交易，并以周块估计收益不确定性。
def write_diagnostics(table: pl.DataFrame, config: dict, output: Path) -> None:
    """输出分组CSV、最差交易与完整日序列的七日块重采样，诊断不产生新交易规则。"""
    for dimension in ['side', 'month', 'signal_strength', 'volatility_regime', 'outcome']:
        table.group_by(['split', dimension]).agg(
            pl.len().alias('trades'),
            (pl.col('net_pnl_usdt') > 0).mean().alias('win_rate'),
            pl.col('gross_pnl_usdt').sum(), pl.col('fees_usdt').sum(), pl.col('funding_paid_usdt').sum(),
            pl.col('net_pnl_usdt').sum(), pl.col('net_pnl_usdt').min().alias('worst_trade_usdt'),
            pl.col('profit_giveback').sum(), pl.col('recovered_after_50bps_adverse').sum(),
        ).sort(['split', dimension]).write_csv(output / f'by_{dimension}.csv')
    table.sort('net_pnl_usdt').write_csv(output / 'trade_cases.csv')
    uncertainty = []
    generator = np.random.default_rng(17)
    for split in ['val_dev', 'val_select']:
        daily = pl.read_csv(Path(config['profit_output']) / 'cases' / f'{config["case"]}_{split}.daily.csv')
        returns = daily['net_pnl_usdt'].to_numpy()
        # 从完整日历抽取连续七日块，含零交易日，保留跨日持仓造成的局部依赖。
        starts = generator.integers(0, len(returns), size=(10000, (len(returns) + 6) // 7))
        positions = (starts[:, :, None] + np.arange(7)) % len(returns)
        samples = returns[positions.reshape(10000, -1)[:, :len(returns)]].sum(axis=1)
        actual = table.filter(pl.col('split') == split)
        uncertainty.append({
            'split': split, 'net_pnl_usdt': float(returns.sum()),
            'bootstrap_p05_usdt': float(np.quantile(samples, .05)),
            'bootstrap_p95_usdt': float(np.quantile(samples, .95)),
            'bootstrap_positive_fraction': float(np.mean(samples > 0)),
            'profit_giveback_trades': int(actual['profit_giveback'].sum()),
            'recovered_after_50bps_adverse_trades': int(actual['recovered_after_50bps_adverse'].sum()),
        })
    write_json_atomic(output / 'uncertainty.json', {
        'method': '10000 circular seven-day block resamples, fixed seed 17, historical diagnosis',
        'results': uncertainty,
    })


# 【非核心】按空仓边界归并原生成交，部分开平仓合计数量与费用，资金费沿用实际现金流。
def native_round_trips(run: Path) -> pl.DataFrame:
    """读取一个已完成区间的成交和资金费，输出按实际交易所时间归并的完整往返。"""
    fills = pl.read_parquet(run / 'fills.parquet').sort('exchange_timestamp_ns')
    payments = json.loads((run / 'funding_cashflows.json').read_text())['records']
    if fills.is_empty():
        return pl.DataFrame()
    positions = fills['position'].to_numpy()
    identifiers = np.cumsum(np.r_[True, np.abs(positions[:-1]) < 1e-10])
    records = []
    for identifier in np.unique(identifiers):
        trade = fills.filter(pl.Series(identifiers == identifier))
        first = trade.row(0, named=True)
        last = trade.row(-1, named=True)
        opening = trade.filter(pl.col('side') == first['side'])
        entry_notional = float(opening.select((pl.col('price') * pl.col('quantity')).sum()).item())
        gross = float(trade.select((-pl.col('side') * pl.col('price') * pl.col('quantity')).sum()).item())
        fees = float(trade['fee'].sum())
        # 无资金费时也保持浮点金额，保证各区间账本列类型一致。
        funding = sum((payment['funding_payment_usdt'] for payment in payments
                       if first['exchange_timestamp_ns'] < payment['timestamp_ns'] <= last['exchange_timestamp_ns']), 0.0)
        records.append({
            'native_trade_id': int(identifier), 'entry_time_s': first['exchange_timestamp_ns'] / SECOND,
            'exit_time_s': last['exchange_timestamp_ns'] / SECOND, 'side': first['side'],
            'quantity_btc': float(opening['quantity'].sum()), 'entry_notional_usdt': entry_notional,
            'gross_pnl_usdt': gross, 'fees_usdt': fees, 'funding_paid_usdt': funding,
            'net_pnl_usdt': gross - fees - funding,
        })
    return pl.DataFrame(records)


# 【核心逻辑380】对齐完整原生回放与快照交易，分别保留匹配、额外和未成交机会。
def compare_native_execution(filename: Path, native_output: Path) -> None:
    """只在全部冻结区间完成后执行，按同方向且入场相差不超过一秒匹配，不强行配对。"""
    _, config, output = context(filename)
    blocks = json.loads((native_output / 'blocks.json').read_text())['blocks']
    native_frames = []
    for block in blocks:
        result = json.loads((native_output / 'block_results' / f'{block["id"]}.json').read_text())
        frame = native_round_trips(native_output / 'runs' / block['id'])
        if frame.is_empty():
            if abs(result['net_pnl_usdt']) > 1e-7:
                raise ValueError('无成交区间存在非零净收益')
            continue
        if abs(frame['net_pnl_usdt'].sum() - result['net_pnl_usdt']) > 1e-7:
            raise ValueError('原生逐笔往返与区间账本不一致')
        native_frames.append(frame.with_columns(pl.lit(block['split']).alias('split'), pl.lit(block['id']).alias('block')))
    native = pl.concat(native_frames)
    native.write_parquet(output / 'native_trades.parquet', compression='zstd')
    comparisons = []
    for split in ['val_dev', 'val_select']:
        snapshot = pl.read_parquet(Path(config['profit_output']) / 'cases' / f'{config["case"]}_{split}.trades.parquet')
        matched_ids = set()
        for trade in native.filter(pl.col('split') == split).iter_rows(named=True):
            candidates = snapshot.filter(
                (pl.col('side') == trade['side'])
                & ((pl.col('entry_time_s') - trade['entry_time_s']).abs() <= 1.)
                & ~pl.col('trade_id').is_in(list(matched_ids)),
            )
            record = {
                'split': split, 'block': trade['block'], 'native_trade_id': trade['native_trade_id'],
                'snapshot_trade_id': None, 'status': 'native_only',
                'native_net_usdt': trade['net_pnl_usdt'], 'snapshot_net_usdt': None,
                'entry_delay_seconds': None, 'exit_delay_seconds': None, 'net_difference_usdt': None,
            }
            if candidates.height > 1:
                raise ValueError('同一原生入场对应多个快照交易')
            if candidates.height == 1:
                reference = candidates.row(0, named=True)
                matched_ids.add(reference['trade_id'])
                record.update(
                    snapshot_trade_id=reference['trade_id'], status='matched',
                    snapshot_net_usdt=reference['net_pnl_usdt'],
                    entry_delay_seconds=trade['entry_time_s'] - reference['entry_time_s'],
                    exit_delay_seconds=trade['exit_time_s'] - reference['exit_time_s'],
                    net_difference_usdt=trade['net_pnl_usdt'] - reference['net_pnl_usdt'],
                )
            comparisons.append(record)
        for reference in snapshot.filter(~pl.col('trade_id').is_in(list(matched_ids))).iter_rows(named=True):
            comparisons.append({
                'split': split, 'block': None, 'native_trade_id': None, 'snapshot_trade_id': reference['trade_id'],
                'status': 'snapshot_only', 'native_net_usdt': None, 'snapshot_net_usdt': reference['net_pnl_usdt'],
                'entry_delay_seconds': None, 'exit_delay_seconds': None, 'net_difference_usdt': None,
            })
    pl.DataFrame(comparisons).write_csv(output / 'execution_comparison.csv')
    summarize_native_risk(config, output, native_output, blocks, native)
    write_json_atomic(output / 'execution_completed.json', {'status': 'success', 'native_round_trips': native.height})


# 【核心逻辑381】在完全相同的交易窗口比较方向贡献，所有方向均用十档盘口和相同费率执行。
def compare_trade_directions(filename: Path) -> None:
    """固定已发生的入场、退出及数量，比较模型方向、恒多、恒空和反向；不重新选交易时刻。"""
    from mm_research.multihorizon.inventory import walk_levels

    _, config, output = context(filename)
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    segments = json.loads((source / 'e0/splits.json').read_text())['segments']
    origin = manifest['start_timestamp_ns'] / SECOND
    records = []
    for split in ['val_dev', 'val_select']:
        trades = pl.read_parquet(Path(config['profit_output']) / 'cases' / f'{config["case"]}_{split}.trades.parquet')
        books = np.load(source / f'e3/{split}_books.npy', mmap_mode='r')
        for trade in trades.iter_rows(named=True):
            entry_row = int(np.floor(trade['entry_time_s'] - origin)) - segments[split]['start_row']
            exit_row = int(np.floor(trade['exit_time_s'] - origin)) - segments[split]['start_row']
            entry_book = books[entry_row]
            exit_book = books[exit_row]
            quantity = trade['quantity_btc']
            directions = {'model': trade['side'], 'always_long': 1., 'always_short': -1., 'reversed': -trade['side']}
            for name, side in directions.items():
                entry_offset = 20 if side > 0 else 0
                exit_offset = 0 if side > 0 else 20
                opened, entry_price = walk_levels(
                    entry_book[entry_offset:entry_offset + 10], entry_book[entry_offset + 10:entry_offset + 20], quantity, .001,
                )
                closed, exit_price = walk_levels(
                    exit_book[exit_offset:exit_offset + 10], exit_book[exit_offset + 10:exit_offset + 20], quantity, .001,
                )
                full_quantity = abs(opened - quantity) < 1e-10 and abs(closed - quantity) < 1e-10
                gross = side * quantity * (exit_price - entry_price) if full_quantity else None
                fees = quantity * (entry_price + exit_price) * .0005 if full_quantity else None
                funding = trade['funding_paid_usdt'] * side / trade['side'] if full_quantity else None
                net = gross - fees - funding if full_quantity else None
                if name == 'model' and (not full_quantity or abs(net - trade['net_pnl_usdt']) > 1e-6):
                    raise ValueError('固定窗口方向对照未复现原交易账本')
                records.append({
                    'split': split, 'trade_id': trade['trade_id'], 'direction_rule': name, 'side': side,
                    'full_quantity': full_quantity, 'gross_pnl_usdt': gross, 'fees_usdt': fees,
                    'funding_paid_usdt': funding, 'net_pnl_usdt': net,
                })
    table = pl.DataFrame(records)
    table.write_csv(output / 'direction_comparison_trades.csv')
    table.group_by(['split', 'direction_rule']).agg(
        pl.len().alias('requested_trades'), pl.col('full_quantity').sum().alias('fully_executable_trades'),
        pl.col('net_pnl_usdt').sum().alias('net_pnl_executable_trades_usdt'),
        pl.col('gross_pnl_usdt').sum(), pl.col('fees_usdt').sum(), pl.col('funding_paid_usdt').sum(),
    ).sort(['split', 'direction_rule']).write_csv(output / 'direction_comparison.csv')



# 【核心逻辑382】将实际资金费加入原生权益曲线，跨空仓区间连续统计回撤、逐日收益及月度归因。
def summarize_native_risk(config: dict, output: Path, native_output: Path, blocks: list[dict], trades: pl.DataFrame) -> None:
    """只保留日级权益和汇总风险，区间内使用完整已记录权益曲线计算回撤。"""
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    segments = json.loads((source / 'e0/splits.json').read_text())['segments']
    summaries = []
    daily_frames = []
    for split in ['val_dev', 'val_select']:
        peak = 10000.
        drawdown = 0.
        daily = []
        for block in blocks:
            if block['split'] != split:
                continue
            run = native_output / 'runs' / block['id']
            payments = json.loads((run / 'funding_cashflows.json').read_text())['records']
            funding = pl.DataFrame(payments).select(
                pl.col('timestamp_ns').alias('local_timestamp_ns'),
                pl.col('funding_payment_usdt').cum_sum().alias('funding_paid'),
            ).sort('local_timestamp_ns')
            equity = pl.read_parquet(run / 'equity.parquet', columns=['local_timestamp_ns', 'equity']).sort('local_timestamp_ns')
            equity = equity.join_asof(funding, on='local_timestamp_ns', strategy='backward').with_columns(
                (pl.col('equity') - pl.col('funding_paid').fill_null(0.)).alias('equity'),
            )
            values = equity['equity'].to_numpy()
            result = json.loads((native_output / 'block_results' / f'{block["id"]}.json').read_text())
            if abs(values[-1] - result['ending_account_equity_usdt']) > 1e-6:
                raise ValueError('原生权益曲线与含资金费的区间结算不一致')
            running_peak = np.maximum.accumulate(np.r_[peak, values])[1:]
            drawdown = max(drawdown, float(np.max(running_peak - values)))
            peak = float(running_peak[-1])
            daily.append(equity.with_columns(
                pl.col('local_timestamp_ns').cast(pl.Datetime('ns')).dt.date().alias('date'),
            ).group_by('date').agg(pl.col('equity').last()))
        bounds = segments[split]
        first = bounds['start_row'] // 86400
        last = (bounds['stop_row'] - 1) // 86400
        calendar = pl.DataFrame({'date': manifest['dates'][first:last + 1]}).with_columns(pl.col('date').str.to_date())
        observed = pl.concat(daily).group_by('date').agg(pl.col('equity').last())
        curve = calendar.join(observed, on='date', how='left').sort('date').with_columns(
            pl.col('equity').forward_fill().fill_null(10000.),
        ).with_columns(
            (pl.col('equity') - pl.col('equity').shift(1).fill_null(10000.)).alias('net_pnl_usdt'),
            pl.lit(split).alias('split'),
        )
        if abs(curve['net_pnl_usdt'].sum() - trades.filter(pl.col('split') == split)['net_pnl_usdt'].sum()) > 1e-6:
            raise ValueError('原生完整日历收益与逐笔往返不一致')
        daily_frames.append(curve)
        summaries.append({
            'split': split, 'net_pnl_usdt': float(curve['net_pnl_usdt'].sum()),
            'max_drawdown_usdt': drawdown, 'ending_equity_usdt': float(curve['equity'][-1]),
            'excluding_best_day_usdt': float(curve['net_pnl_usdt'].sum() - curve['net_pnl_usdt'].max()),
            'round_trips': trades.filter(pl.col('split') == split).height,
        })
    pl.concat(daily_frames).write_csv(output / 'native_daily.csv')
    write_json_atomic(output / 'native_risk.json', {'basis': 'recorded_native_equity_plus_actual_funding', 'results': summaries})
