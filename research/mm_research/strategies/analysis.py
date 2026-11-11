"""从实际成交核对现金、费用和清算权益，输出策略收益与成交后Markout。"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from mm_research.data.storage import write_json_atomic


# 【非核心】按账本明确字段读取CSV，零成交实验也保留稳定的数值schema。
def read_execution_frames(run: Path) -> dict[str, pl.DataFrame]:
    """返回订单、成交、盯市权益、清算权益及IOC用途五张原生输出表。"""
    float_fields = {
        'orders': ['price', 'quantity', 'leaves_quantity', 'exec_price', 'exec_quantity'],
        'fills': ['price', 'quantity', 'fee', 'position', 'balance', 'average_cost', 'realized_gross', 'mid', 'net_pnl'],
        'equity': ['mid', 'position', 'balance', 'fees', 'inventory_value', 'realized_gross', 'unrealized', 'net_pnl',
                   'equity', 'reserved_buy', 'reserved_sell', 'running_max_drawdown'],
        'liquidation_equity': ['mid', 'bid', 'ask', 'position', 'net_pnl_liquidated'],
        'ioc_actions': [],
    }
    frames = {}
    for name, fields in float_fields.items():
        schema = {field: pl.Float64 for field in fields}
        schema['local_timestamp_ns'] = pl.Int64
        if name in ['orders', 'fills', 'ioc_actions']:
            schema['order_id'] = pl.UInt64
        if name in ['orders', 'fills']:
            schema.update(exchange_timestamp_ns=pl.Int64, side=pl.Int8, maker=pl.Int8)
        if name == 'fills':
            schema['fill_id'] = pl.UInt64
        if name == 'liquidation_equity':
            schema['book_fresh'] = pl.Int8
        if name == 'ioc_actions':
            schema['purpose'] = pl.String
        frames[name] = pl.read_csv(run / f'{name}.csv', schema_overrides=schema, null_values='nan')
    return frames


# 【核心逻辑246】仅在成交发生后计算Markout，截止时刻之后的数据不足则保留空值。
def add_fill_markouts(fills: pl.DataFrame, liquidation: pl.DataFrame) -> pl.DataFrame:
    """交易所成交时刻加h和20ms，在100ms可见盘口上向后取价，误差窗口至多100ms。"""
    mid = liquidation.select('local_timestamp_ns', 'mid').unique('local_timestamp_ns', keep='last').sort('local_timestamp_ns')
    result = fills.sort('exchange_timestamp_ns')
    for horizon in [.1, 1, 5, 10, 30, 60, 120, 180, 300]:
        suffix = '100ms' if horizon == .1 else f'{horizon}s'
        target = pl.col('exchange_timestamp_ns') + int(horizon * 1_000_000_000) + 20_000_000
        result = result.with_columns(target.alias('_mark_time')).join_asof(
            mid.rename({'local_timestamp_ns': '_mid_time', 'mid': '_future_mid'}),
            left_on='_mark_time', right_on='_mid_time', strategy='backward', tolerance=100_000_000,
        ).with_columns(pl.when(pl.col('_mark_time') <= mid['local_timestamp_ns'].max())
                       .then(pl.col('side') * (pl.col('_future_mid') / pl.col('price') - 1) * 10000)
                       .otherwise(None).alias(f'markout_{suffix}_bps')).drop('_mark_time', '_mid_time', '_future_mid')
    return result


# 【核心逻辑247】资金费按结算时刻的实际交易所成交仓位计入现金流，缺少费率时明确保留未覆盖状态。
def funding_cashflows(output: Path, fills: pl.DataFrame, date: str) -> tuple[pl.DataFrame, str]:
    """只接受已归档的交易所资金费记录，资金费缺失时不生成推定金额。"""
    file = output / f'funding_{date}.json'
    schema = {'timestamp_ns': pl.Int64, 'position': pl.Float64, 'funding_payment': pl.Float64,
              'funding_payment_low': pl.Float64, 'funding_payment_high': pl.Float64}
    if not file.exists():
        return pl.DataFrame(schema=schema), 'unavailable'
    funding = json.loads(file.read_text())
    records = []
    for event in funding['records']:
        timestamp = int(event['fundingTime']) * 1_000_000
        position = fills.filter(pl.col('exchange_timestamp_ns') <= timestamp).select(
            (pl.col('side') * pl.col('quantity')).sum()).item()
        payment = position * float(event['markPrice']) * float(event['fundingRate'])
        alternatives = [position * float(event[price]) * float(event['fundingRate'])
                        for price in ['markPriceLow', 'markPriceHigh']]
        records.append({'timestamp_ns': timestamp, 'position': position, 'funding_payment': payment,
                        'funding_payment_low': min(alternatives), 'funding_payment_high': max(alternatives)})
    return pl.DataFrame(records, schema=schema), 'official_archive_with_mark_interval'


# 【核心逻辑248】从实际平仓回合提取持有时间，库存风险不由预测持仓替代。
def completed_holding_times(fills: pl.DataFrame) -> np.ndarray:
    """返回从空仓进入到再次空仓的实际交易所时间长度，单位秒。"""
    opened = None
    durations = []
    for row in fills.sort('exchange_timestamp_ns').iter_rows(named=True):
        if opened is None:
            opened = row['exchange_timestamp_ns']
        if abs(row['position']) < 1e-10:
            durations.append((row['exchange_timestamp_ns'] - opened) / 1e9)
            opened = None
    return np.asarray(durations, dtype=np.float64)


# 【核心逻辑245】独立成交现金流必须与原生账本相等，最终收益以已实际平仓的现金计算。
def analyze_strategy_case(root: Path, config: dict, case: dict, run: Path) -> dict:
    """保存费用、成交率、清算回撤和Markout，核对成功后用Parquet替换同内容CSV。"""
    frames = read_execution_frames(run)
    fills = frames['fills']
    orders = frames['orders']
    summary = json.loads((run / 'summary.json').read_text())
    expected_cash = -fills.select((pl.col('side') * pl.col('price') * pl.col('quantity')).sum()).item()
    expected_fee = fills.select((pl.col('price') * pl.col('quantity')
        * pl.when(pl.col('maker') == 1).then(config['maker_fee']).otherwise(config['taker_fee'])).sum()).item()
    if abs(expected_cash - summary['balance']) > 1e-7 or abs(expected_fee - summary['fees']) > 1e-7:
        raise ValueError('independent trading cash or fee does not reconcile')
    if abs(summary['position']) > 1e-10 or summary['final_working_orders'] != 0:
        raise ValueError('strategy did not end flat with all orders resolved')
    manifest = json.loads((root / config['source']).read_text())
    if summary['local_events'] != manifest['logical_events'] or summary['exchange_events'] != manifest['logical_events']:
        raise ValueError('replay did not consume the full native day')
    fills = add_fill_markouts(fills, frames['liquidation_equity'])
    fills = fills.join(frames['ioc_actions'].select('order_id', 'purpose'), on='order_id', how='left')
    forced = fills.filter(pl.col('purpose').is_in(['max_hold', 'final_liquidation']))
    maker = fills.filter(pl.col('maker') == 1)
    maker_submissions = orders.filter((pl.col('kind') == 'submit') & (pl.col('order_id') < 1_000_000_000_000)).height
    funding, funding_status = funding_cashflows(root / config['output'], fills, config['date'])
    funding_amount = funding['funding_payment'].sum() if funding_status == 'official_archive_with_mark_interval' else None
    liquidation = frames['liquidation_equity']
    payments = np.zeros(liquidation.height)
    for event in funding.iter_rows(named=True):
        payments += np.where(liquidation['local_timestamp_ns'].to_numpy() >= event['timestamp_ns'], event['funding_payment'], 0.)
    pnl = liquidation['net_pnl_liquidated'].to_numpy() - payments
    valid_pnl = pnl[liquidation['book_fresh'].to_numpy() == 1]
    equity = config['initial_capital'] + valid_pnl
    peaks = np.maximum.accumulate(np.r_[config['initial_capital'], equity])[1:]
    durations = completed_holding_times(fills)
    result = {
        **case, 'date': config['date'], 'fills': fills.height, 'maker_fills': maker.height,
        'native_replay_executed': True, 'result_origin': 'native_event_replay',
        'taker_fills': fills.height - maker.height, 'maker_submissions': maker_submissions,
        'maker_order_fill_rate': maker['order_id'].n_unique() / maker_submissions if maker_submissions else None,
        'cancel_requests': summary['cancel_requests'], 'cancel_to_fill': summary['cancel_requests'] / fills.height if fills.height else None,
        'turnover_usdt': summary['turnover'], 'gross_trading_pnl_usdt': expected_cash,
        'trading_fees_usdt': expected_fee, 'net_trading_pnl_usdt': expected_cash - expected_fee,
        'zero_fee_same_fills_pnl_usdt': expected_cash, 'funding_status': funding_status,
        'funding_payment_usdt': funding_amount,
        'net_pnl_after_funding_usdt': expected_cash - expected_fee - funding_amount if funding_amount is not None else None,
        'net_bps_per_turnover': (expected_cash - expected_fee) / summary['turnover'] * 10000 if summary['turnover'] else None,
        'max_liquidation_drawdown_usdt': float(np.max(peaks - equity)),
        'max_abs_position_btc': summary['max_abs_position'], 'final_position_btc': summary['position'],
        'forced_exit_fees_usdt': forced['fee'].sum(),
        'forced_exit_price_cost_vs_mid_usdt': forced.select((pl.col('side') * (pl.col('price') - pl.col('mid')) * pl.col('quantity')).sum()).item(),
        'mean_holding_seconds': float(durations.mean()) if len(durations) else None,
        'p95_holding_seconds': float(np.quantile(durations, .95)) if len(durations) else None,
        'local_events': summary['local_events'], 'exchange_events': summary['exchange_events'],
        'replay_seconds': summary['elapsed_seconds'], 'ledger_reconciled': True,
    }
    for horizon in [.1, 1, 5, 10, 30, 60, 120, 180, 300]:
        suffix = '100ms' if horizon == .1 else f'{horizon}s'
        result[f'maker_markout_{suffix}_bps'] = maker[f'markout_{suffix}_bps'].mean()
    frames['fills'] = fills
    frames['liquidation_equity'] = liquidation.with_columns(pl.Series('net_pnl_with_funding', pnl))
    funding.write_parquet(run / 'funding_cashflows.parquet')
    for name, table in frames.items():
        table.write_parquet(run / f'{name}.parquet', compression='zstd')
    write_json_atomic(run / 'execution_metrics.json', result)
    for name in frames:
        (run / f'{name}.csv').unlink()
    return result


# 【核心逻辑249】汇总实际完成的策略账本，失败和资金费覆盖状态均保留在结果表。
def build_strategy_report(root: Path, config: dict) -> list[dict]:
    """收益与模型评价分别归档，单日结果不生成年化Sharpe。"""
    output = root / config['output']
    settle_funding_results(output, config)
    records = [json.loads(path.read_text()) for path in sorted((output / 'runs').glob('*/execution_metrics.json'))]
    table = pl.DataFrame(records, infer_schema_length=None)
    table.write_parquet(output / 'strategy_metrics.parquet')
    table.write_csv(output / 'strategy_metrics.csv')
    return records


# 【核心逻辑250】从官方费率和标记价格档案生成单日结算输入，分钟估值误差显式保存。
def prepare_funding_archive(output: Path, date: str) -> None:
    """保留calc_time，使用对应分钟开盘标记价及包含前一分钟收盘的价格区间。"""
    rates = pl.read_csv(output / 'funding_2026-10.csv')
    marks = pl.read_csv(output / f'mark_price_{date}.csv')
    start = int(np.datetime64(date, 'ms').astype(np.int64))
    rates = rates.filter(pl.col('calc_time').is_between(start, start + 86400000, closed='left'))
    records = []
    for event in rates.iter_rows(named=True):
        minute = event['calc_time'] // 60000 * 60000
        mark = marks.filter(pl.col('open_time') == minute).row(0, named=True)
        previous = marks.filter(pl.col('open_time') == minute - 60000)
        low = mark['low']
        high = mark['high']
        if previous.height:
            low = min(low, previous['close'][0])
            high = max(high, previous['close'][0])
        records.append({'fundingTime': event['calc_time'], 'fundingRate': event['last_funding_rate'],
                        'fundingIntervalHours': event['funding_interval_hours'], 'markPrice': mark['open'],
                        'markPriceLow': low, 'markPriceHigh': high})
    if len(records) != 3:
        raise ValueError('2026-10-21 official funding archive does not contain the expected three settlements')
    write_json_atomic(output / f'funding_{date}.json', {
        'source': 'https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2026-10.zip',
        'mark_source': f'https://data.binance.vision/data/futures/um/daily/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-{date}.zip',
        'valuation': 'official_1m_open_proxy_with_price_bounds', 'settlement_clock': 'official_calc_time', 'records': records,
    })


# 【核心逻辑251】费用档案补齐后统一结算已完成账本，原生成交与现金流无需重新回放。
def settle_funding_results(output: Path, config: dict) -> None:
    """资金费估值和清算回撤统一更新，证明从未下单的组合仍保持明确的非回放状态。"""
    for file in sorted((output / 'runs').glob('*/execution_metrics.json')):
        result = json.loads(file.read_text())
        if result['result_origin'] == 'no_order_proof':
            continue
        run = file.parent
        fills = pl.read_parquet(run / 'fills.parquet')
        funding, status = funding_cashflows(output, fills, config['date'])
        if status == 'unavailable':
            continue
        payment = funding['funding_payment'].sum()
        result.update(funding_status=status, funding_payment_usdt=payment,
                      net_pnl_after_funding_usdt=result['net_trading_pnl_usdt'] - payment,
                      net_pnl_funding_lower_usdt=result['net_trading_pnl_usdt'] - funding['funding_payment_high'].sum(),
                      net_pnl_funding_upper_usdt=result['net_trading_pnl_usdt'] - funding['funding_payment_low'].sum())
        liquidation = pl.read_parquet(run / 'liquidation_equity.parquet')
        cumulative = np.zeros(liquidation.height)
        for event in funding.iter_rows(named=True):
            cumulative += np.where(liquidation['local_timestamp_ns'].to_numpy() >= event['timestamp_ns'], event['funding_payment'], 0.)
        pnl = liquidation['net_pnl_liquidated'].to_numpy() - cumulative
        observed = config['initial_capital'] + pnl[liquidation['book_fresh'].to_numpy() == 1]
        peaks = np.maximum.accumulate(np.r_[config['initial_capital'], observed])[1:]
        result['max_liquidation_drawdown_usdt'] = float(np.max(peaks - observed))
        liquidation.with_columns(pl.Series('net_pnl_with_funding', pnl)).write_parquet(run / 'liquidation_equity.parquet', compression='zstd')
        funding.write_parquet(run / 'funding_cashflows.parquet')
        write_json_atomic(file, result)
