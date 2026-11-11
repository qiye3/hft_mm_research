"""预测曲线的离散动态规划和带延迟的快照可执行价Taker对照。"""

import io
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile

from numba import njit
import numpy as np
import polars as pl

from mm_research.data.storage import file_locks, write_json_atomic
from mm_research.multihorizon.common import HORIZONS, DAY, SECOND, context, open_dataset
from mm_research.multihorizon.evaluation import prediction_frame, calibrated_curve


# 【核心逻辑271】五档库存的动态规划从实际仓位出发，终点计入平仓成本。
@njit(cache=True)
def optimal_path(increments, variances, risk_lambda, cost_bps, actual_units):
    """返回16区段的最优计划，实际只执行第一段，下一秒重新规划。"""
    states = np.arange(-2., 3.)
    count = len(increments)
    parents = np.zeros((count, 5), dtype=np.int64)
    values = np.empty(5)
    for current in range(5):
        units = states[current]
        values[current] = units * increments[0] - .5 * risk_lambda * variances[0] * units * units - cost_bps * abs(units - actual_units)
    for step in range(1, count):
        next_values = np.empty(5)
        for current in range(5):
            units = states[current]
            best = -np.inf
            predecessor = 0
            for previous in range(5):
                candidate = values[previous] - cost_bps * abs(units - states[previous])
                if candidate > best:
                    best = candidate
                    predecessor = previous
            next_values[current] = best + units * increments[step] - .5 * risk_lambda * variances[step] * units * units
            parents[step, current] = predecessor
        values = next_values
    best = -np.inf
    selected = 0
    for index in range(5):
        candidate = values[index] - cost_bps * abs(states[index])
        if candidate > best:
            best = candidate
            selected = index
    path = np.zeros(count)
    for step in range(count - 1, -1, -1):
        path[step] = states[selected]
        selected = parents[step, selected]
    return path


# 【核心逻辑272】只按到达时刻已存在的前十档逐层成交，不足数量保留为未成交。
@njit(cache=True)
def walk_levels(prices, quantities, requested, quantity_step):
    """返回可执行整手数量与加权均价，绝不通过中间价填满剩余数量。"""
    remaining = requested
    filled = 0.
    notional = 0.
    for level in range(len(prices)):
        available = np.floor((quantities[level] + 1e-12) / quantity_step) * quantity_step
        amount = min(remaining, available)
        if amount > 0 and prices[level] > 0:
            filled += amount
            notional += amount * prices[level]
            remaining -= amount
        if remaining < quantity_step * .5:
            break
    return filled, notional / filled if filled > 0 else 0.


# 【核心逻辑273】在真实已成交仓位上计算P0—P3，现金、费用与逐档清算价值分开累计。
@njit(cache=True)
def simulate_inventory(predictions, books, times, variance, policy, horizon_index, hold_seconds,
                       risk_lambda, buffer_bps, rebalance_seconds, unit_btc, fee, min_notional, quantity_step,
                       max_hold_seconds, initial_capital, funding_times, funding_rates, funding_marks):
    """books为订单到达时最近已发生快照，结果显式标记快照执行估计。"""
    count = len(times)
    position = 0.
    cash = 0.
    fees = 0.
    funding = 0.
    opened_at = -1.
    last_rebalance = -np.inf
    funding_cursor = 0
    records = np.zeros((count, 11))
    fills = np.zeros((count, 7))
    fill_count = 0
    rejected = 0
    margin_rejected = 0
    for row in range(count):
        now = times[row]
        while funding_cursor < len(funding_times) and funding_times[funding_cursor] <= now:
            payment = position * funding_marks[funding_cursor] * funding_rates[funding_cursor]
            cash -= payment
            funding += payment
            funding_cursor += 1
        book = books[row]
        valid_book = np.isfinite(book).all() and book[0] < book[20] and book[0] > 0
        curve = predictions[row]
        actual = position / unit_btc
        target = actual
        if valid_book and np.isfinite(curve[horizon_index]):
            spread_bps = (book[20] - book[0]) / ((book[20] + book[0]) * .5) * 10000
            one_way_cost = fee * 10000 + spread_bps * .5
            alpha = curve[horizon_index]
            direction = 1. if alpha > 0 else (-1. if alpha < 0 else 0.)
            if policy in [0, 1]:
                if abs(actual) < .25:
                    if policy == 0 or abs(alpha) > 2 * one_way_cost + buffer_bps:
                        target = direction
                elif now - opened_at >= hold_seconds:
                    target = 0.
            elif policy == 2:
                raw_target = np.round(alpha / max(risk_lambda * variance[horizon_index], 1e-8))
                candidate = min(2., max(-2., raw_target))
                benefit = (candidate - actual) * alpha - .5 * risk_lambda * variance[horizon_index] * (candidate ** 2 - actual ** 2)
                if benefit > (one_way_cost + buffer_bps) * abs(candidate - actual):
                    target = candidate
            else:
                if np.isfinite(curve).all():
                    increments = np.empty(len(curve))
                    increments[0] = curve[0]
                    increments[1:] = curve[1:] - curve[:-1]
                    target = optimal_path(increments, variance, risk_lambda, one_way_cost + buffer_bps, actual)[0]
        force_exit = (row >= count - 20) or (position != 0 and now - opened_at >= max_hold_seconds)
        if force_exit:
            target = 0.
        if now - last_rebalance < rebalance_seconds and not force_exit:
            target = actual
        requested = target * unit_btc - position
        if position * target < 0:
            requested = -position
        if valid_book and abs(requested) >= quantity_step * .5:
            side = 1 if requested > 0 else -1
            if side > 0:
                quantity, price = walk_levels(book[20:30], book[30:40], abs(requested), quantity_step)
            else:
                quantity, price = walk_levels(book[:10], book[10:20], abs(requested), quantity_step)
            if quantity * price < min_notional and position * side >= 0:
                quantity = 0.
                rejected += 1
            equity_before = initial_capital + cash + position * ((book[0] + book[20]) * .5)
            margin_after = abs(position + side * quantity) * price
            if position * side >= 0 and equity_before < margin_after + quantity * price * fee:
                quantity = 0.
                margin_rejected += 1
            if quantity > 0:
                previous_position = position
                position += side * quantity
                cash -= side * quantity * price
                charged = quantity * price * fee
                fees += charged
                cash -= charged
                if previous_position == 0 or previous_position * position < 0:
                    opened_at = now
                if abs(position) < 1e-10:
                    position = 0.
                    opened_at = -1.
                last_rebalance = now
                fills[fill_count] = np.array([now, side, quantity, price, charged, position, 1. if force_exit else 0.])
                fill_count += 1
        equity = np.nan
        if valid_book:
            if position >= 0:
                closing, close_price = walk_levels(book[:10], book[10:20], abs(position), quantity_step)
            else:
                closing, close_price = walk_levels(book[20:30], book[30:40], abs(position), quantity_step)
            if abs(position) < 1e-10:
                equity = cash
            elif abs(closing - abs(position)) < 1e-10:
                equity = cash + position * close_price - closing * close_price * fee
        records[row] = np.array([now, target, position / unit_btc, position, cash, fees, funding, equity,
                                 rejected, 1. if valid_book else 0., margin_rejected])
    return records, fills[:fill_count], position


# 【非核心】将完整验证网格匹配到实际到达时刻的原始盘口，保留十档价量与缺失。
def execution_books(root: Path, output: Path, config: dict, segment: str) -> np.ndarray:
    """只使用exchange_timestamp不晚于到达时刻的快照，误差由快照采样频率决定。"""
    destination = output / 'e3' / f'{segment}_books.npy'
    if destination.exists():
        return np.load(destination, mmap_mode='r')
    manifest, splits, matrix, targets, market = open_dataset(output)
    bounds = splits['segments'][segment]
    count = bounds['stop_row'] - bounds['start_row']
    books = np.lib.format.open_memmap(destination, mode='w+', dtype='float64', shape=(count, 40))
    fields = [f'{side}_{field}_{level}' for side in ['bid', 'ask'] for field in ['price', 'quantity'] for level in range(1, 11)]
    latency = config['prediction_latency_ns'] + config['execution']['entry_latency_ns']
    for begin in range(bounds['start_row'], bounds['stop_row'], DAY):
        date = manifest['dates'][begin // DAY]
        timestamps = manifest['start_timestamp_ns'] + np.arange(begin, begin + DAY) * SECOND + latency
        snapshots = pl.scan_parquet(root / config['standard_folder'] / 'snapshots' / f'{date}.parquet').select('exchange_timestamp_ns', *fields).collect(engine='streaming')
        snapshots = snapshots.sort('exchange_timestamp_ns').unique('exchange_timestamp_ns', keep='last', maintain_order=True)
        table = pl.DataFrame({'arrival_ts_ns': timestamps}).join_asof(snapshots,
            left_on='arrival_ts_ns', right_on='exchange_timestamp_ns', strategy='backward', tolerance=1_000_000_000)
        books[begin - bounds['start_row']:begin - bounds['start_row'] + DAY] = table.select(fields).to_numpy()
    books.flush()
    return books


# 【非核心】取得官方月度资金费及分钟标记价，资金费估值保留来源与分钟代理口径。
def funding_inputs(output: Path, manifest: dict, start_row: int, stop_row: int) -> tuple:
    """只缓存本阶段所需月份，网络失败明确报错，不把未知资金费当零。"""
    folder = output / 'e3/funding'
    folder.mkdir(parents=True, exist_ok=True)
    begin = manifest['start_timestamp_ns'] // SECOND + start_row
    end = manifest['start_timestamp_ns'] // SECOND + stop_row
    months = sorted(set(date[:7] for date in manifest['dates'][start_row // DAY:stop_row // DAY]))
    times = []
    rates = []
    marks = []
    for month in months:
        rate_file = folder / f'{month}_funding.zip'
        mark_file = folder / f'{month}_mark.zip'
        urls = {
            rate_file: f'https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{month}.zip',
            mark_file: f'https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-{month}.zip',
        }
        for file, url in urls.items():
            with file_locks([file]):
                if not file.exists():
                    with urlopen(url, timeout=90) as response:
                        content = response.read()
                    with zipfile.ZipFile(io.BytesIO(content)) as archive:
                        if len(archive.namelist()) != 1:
                            raise ValueError('官方资金费档案成员不唯一')
                    temporary = file.with_suffix('.part')
                    temporary.write_bytes(content)
                    temporary.replace(file)
        with zipfile.ZipFile(rate_file) as archive:
            table = pl.read_csv(archive.open(archive.namelist()[0]),
                                schema_overrides={'calc_time': pl.Int64, 'last_funding_rate': pl.Float64})
        price = read_mark_prices(folder, month)
        price_lookup = dict(zip(price['open_time'].to_list(), price['open'].to_list()))
        for event in table.iter_rows(named=True):
            seconds = event['calc_time'] / 1000
            if begin <= seconds < end:
                times.append(seconds)
                rates.append(event['last_funding_rate'])
                marks.append(price_lookup[event['calc_time'] // 60000 * 60000])
    ordering = np.argsort(times)
    return np.array(times)[ordering], np.array(rates)[ordering], np.array(marks)[ordering]


# 【非核心】合并官方月档与已明确归档的缺失日官方日档，价格不插值且重叠时间明确报错。
def read_mark_prices(folder: Path, month: str) -> pl.DataFrame:
    """日档用于月档真实缺日的来源补充，保留独立原始ZIP和来源记录。"""
    files = [folder / f'{month}_mark.zip', *sorted(folder.glob(f'{month}-??_mark_daily.zip'))]
    frames = []
    schema = {'open_time': pl.Int64, 'open': pl.Float64, 'high': pl.Float64,
              'low': pl.Float64, 'close': pl.Float64}
    for file in files:
        with zipfile.ZipFile(file) as archive:
            frames.append(pl.read_csv(archive.open(archive.namelist()[0]),
                                      columns=list(schema), schema_overrides=schema))
    prices = pl.concat(frames)
    if prices['open_time'].n_unique() != prices.height:
        raise ValueError('官方标记价月档与补充日档存在重叠时间')
    return prices.sort('open_time')


# 【非核心】登记有限单参数候选，模型与仓位变化分别归因，避免全笛卡尔积。
def policy_candidates(config: dict) -> list[dict]:
    """P0/P1固定期限，P2/P3先比较风险系数，默认成本与再平衡作为固定对照。"""
    cases = []
    for policy in [0, 1]:
        for hold in config['execution']['fixed_holds_s']:
            index = int(np.flatnonzero(HORIZONS == hold)[0])
            cases.append({'id': f'P{policy}_h{hold}', 'policy': policy, 'horizon_index': index,
                          'hold_seconds': hold, 'risk_lambda': 1., 'buffer_bps': 0., 'rebalance_seconds': 1,
                          'max_hold_seconds': config['execution']['max_hold_s']})
    for policy in [2, 3]:
        for risk in config['execution']['risk_lambdas']:
            cases.append({'id': f'P{policy}_risk{risk}', 'policy': policy, 'horizon_index': 10,
                          'hold_seconds': 300, 'risk_lambda': risk, 'buffer_bps': 0., 'rebalance_seconds': 1,
                          'max_hold_seconds': config['execution']['max_hold_s']})
    return cases


# 【非核心】先冻结风险系数，再按预登记的单参数候选比较期限、成本缓冲和再平衡。
def refine_inventory_candidates(initial_metrics: list[dict], initial_cases: list[dict], config: dict) -> list[dict]:
    """只读取val_dev初始对照，不扩展笛卡尔积或根据val_select反复改规则。"""
    refined = []
    for policy in [2, 3]:
        scores = [record for record in initial_metrics if record['policy'] == policy]
        winner = max(scores, key=lambda record: record['net_pnl_usdt'])
        base = next(case for case in initial_cases if case['id'] == winner['id'])
        if policy == 2:
            for horizon in [10, 30, 120, 300]:
                case = dict(base)
                case.update(id=f'P2_h{horizon}_risk{base["risk_lambda"]}',
                            horizon_index=int(np.flatnonzero(HORIZONS == horizon)[0]))
                refined.append(case)
        else:
            for buffer in [.5, 1.]:
                case = dict(base)
                case.update(id=f'P3_buffer{buffer}_risk{base["risk_lambda"]}', buffer_bps=buffer)
                refined.append(case)
            for change, value in [('rebalance_seconds', 5), ('max_hold_seconds', 600)]:
                case = dict(base)
                case.update(id=f'P3_{change}{value}_risk{base["risk_lambda"]}')
                case[change] = value
                refined.append(case)
    return refined


# 【核心逻辑274】先在val_dev比较有限候选，再用冻结候选评价val_select，完整账本仅保存选定对照。
def run_inventory(filename: Path) -> None:
    """E3为快照逐档Taker估计，事件级Maker收益由E4另行验证。"""
    root, config, output = context(filename)
    manifest, splits, matrix, targets, market = open_dataset(output)
    folder = output / 'e3'
    folder.mkdir(exist_ok=True)
    selected = json.loads((output / 'e1/selected_models.json').read_text())['models']
    all_metrics = []
    selections = {}
    cases_by_model = {}
    settings = config['execution']
    for model in selected:
        cases = policy_candidates(config)
        initial_count = len(cases)
        calibration = json.loads((output / 'e1/calibration' / f'{model}.json').read_text())
        covariance = np.array(calibration['increment_error_covariance_bps2'])
        increment_variance = np.maximum(np.diag(covariance), 1e-8)
        cumulative_variance = np.maximum(np.cumsum(covariance, axis=0).cumsum(axis=1).diagonal(), 1e-8)
        for segment in ['val_dev', 'val_select']:
            frame = prediction_frame(output, model, segment)
            rows = frame['row'].to_numpy()
            prediction = calibrated_curve(frame, calibration)
            books = execution_books(root, output, config, segment)
            times = (frame['ready_ts_ns'].to_numpy() + settings['entry_latency_ns']) / SECOND
            funding_times, rates, marks = funding_inputs(output, manifest, int(rows[0]), int(rows[-1]) + 1)
            candidates = cases if segment == 'val_dev' else [case for case in cases if case['id'] in selections[model].values()]
            model_metrics = []
            for case in candidates:
                destination = folder / 'runs' / model / segment / case['id']
                destination.mkdir(parents=True, exist_ok=True)
                file = destination / 'metrics.json'
                if file.exists():
                    record = json.loads(file.read_text())
                else:
                    variance = increment_variance if case['policy'] == 3 else cumulative_variance
                    ledger, fills, final = simulate_inventory(prediction, books, times, variance,
                        case['policy'], case['horizon_index'], case['hold_seconds'], case['risk_lambda'],
                        case['buffer_bps'], case['rebalance_seconds'], settings['risk_unit_btc'],
                        settings['taker_fee'], settings['minimum_notional_usdt'], settings['quantity_step_btc'],
                        case['max_hold_seconds'], settings['initial_equity_usdt'], funding_times, rates, marks)
                    equity = ledger[:, 7]
                    finite = np.isfinite(equity)
                    ending = float(equity[finite][-1])
                    gross = float(-(fills[:, 1] * fills[:, 2] * fills[:, 3]).sum())
                    charged = float(fills[:, 4].sum())
                    paid_funding = float(ledger[-1, 6])
                    if abs(final) > 1e-10:
                        raise ValueError(f'未完成实际快照平仓 {model}/{segment}/{case["id"]}')
                    if abs(gross - charged - paid_funding - ending) > 1e-6:
                        raise ValueError('E3现金与可执行清算权益不闭合')
                    peaks = np.maximum.accumulate(np.r_[0., equity[finite]])[1:]
                    turnover = float((fills[:, 2] * fills[:, 3]).sum())
                    record = {'model_id': model, 'split': segment, **case,
                              'net_pnl_usdt': ending, 'gross_pnl_usdt': gross, 'fees_usdt': charged,
                              'funding_paid_usdt': paid_funding, 'trade_count': len(fills), 'turnover_usdt': turnover,
                              'pnl_per_turnover_bps': ending / turnover * 10000 if turnover else None,
                              'max_drawdown_usdt': float(np.max(peaks - equity[finite])),
                              'max_abs_position_btc': float(np.max(np.abs(ledger[:, 3]))),
                              'final_position_btc': final, 'rejected_min_notional': int(ledger[-1, 8]),
                              'rejected_margin': int(ledger[-1, 10]), 'margin_budget': 'one_times_notional',
                              'execution_type': 'arrival_time_backward_snapshot_depth_walk',
                              'funding_valuation': 'official_rate_with_minute_mark_open_proxy'}
                    write_json_atomic(file, record)
                    daily = []
                    previous = 0.
                    for day in range(len(rows) // DAY):
                        values = equity[day * DAY:(day + 1) * DAY]
                        valid = values[np.isfinite(values)]
                        if len(valid):
                            current = float(valid[-1])
                            daily.append({'date': manifest['dates'][int(rows[0]) // DAY + day], 'net_pnl_usdt': current - previous})
                            previous = current
                    pl.DataFrame(daily).write_parquet(destination / 'daily_pnl.parquet')
                    if segment == 'val_select':
                        pl.DataFrame(ledger, schema=['timestamp_s', 'target_units', 'actual_units', 'position_btc',
                            'cash_after_costs', 'fees_usdt', 'funding_usdt', 'liquidation_pnl_usdt', 'rejected', 'book_valid', 'margin_rejected'],
                            orient='row').write_parquet(destination / 'positions_equity.parquet', compression='zstd')
                        pl.DataFrame(fills, schema=['fill_timestamp_s', 'side', 'quantity_btc', 'price', 'fee_usdt',
                            'position_after_btc', 'forced_exit'], orient='row').write_parquet(destination / 'trades.parquet', compression='zstd')
                all_metrics.append(record)
                model_metrics.append(record)
                if segment == 'val_dev' and len(model_metrics) == initial_count:
                    cases.extend(refine_inventory_candidates(model_metrics, cases[:initial_count], config))
                print('E3_CASE', model, segment, case['id'], record['net_pnl_usdt'], flush=True)
            if segment == 'val_dev':
                choices = {}
                for policy in range(4):
                    candidates_for_policy = [record for record in model_metrics if record['policy'] == policy]
                    choices[f'P{policy}'] = max(candidates_for_policy, key=lambda record: record['net_pnl_usdt'])['id']
                selections[model] = choices
                cases_by_model[model] = cases
                write_json_atomic(folder / 'parameter_selection.json', {'fitted_on': 'val_dev', 'models': selections})
    all_metrics = [json.loads(file.read_text()) for file in sorted((folder / 'runs').glob('*/*/*/metrics.json'))]
    pl.DataFrame(all_metrics).write_csv(folder / 'policy_metrics.csv')
    write_json_atomic(folder / 'policy_registry.json', cases_by_model)
    (folder / 'inventory_report.md').write_text('# E3动态库存对照｜2026-11-09\n\n'
        'P0/P1比较固定持有，P2比较单期限目标，P3比较16区间动态规划。'
        '每次从实际成交仓位出发，逐档撮合使用到达时已发生的最近快照，缺口不补未来价格。'
        '统一0.002 BTC风险单位、最大两单位，普通Taker 5 bps与官方资金费分钟估值。'
        'val_dev选择参数后在val_select运行冻结候选。P4未启用。'
        '本阶段为快照执行估计，完整事件级执行见E4。\n')
