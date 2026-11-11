"""逐日原生Maker回放、独立固定订单机会与明确退出规则的执行价值诊断。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import datetime as dt
import fcntl
import json
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
import polars as pl
import yaml

from mm_research.data.storage import file_locks, write_json_atomic
from mm_research.multihorizon.common import HORIZONS, DAY, SECOND, context, open_dataset
from mm_research.multihorizon.models import FrozenCurveModel
from mm_research.multihorizon.inventory import funding_inputs, read_mark_prices
from mm_research.strategies.analysis import analyze_strategy_case


# 【核心逻辑284】完整日期重新推理，模型来自该时刻之前的OOF折或冻结Train权重。
def export_day(output: Path, config: dict, model: str, date: str) -> Path:
    """所有86,400秒只由当前时间决定可用性，未来标签和OOF尾部评价裁切均不参与。"""
    folder = output / 'e4/signals' / model
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f'{date}.csv'
    with file_locks([destination]):
        if destination.exists():
            return destination
        manifest, splits, matrix, targets, market = open_dataset(output)
        start = manifest['dates'].index(date) * DAY
        rows = np.arange(start, start + DAY)
        forecast = np.full((DAY, 16), np.nan)
        if model == 'zero':
            forecast[:] = 0.
            intervals = []
        elif start >= splits['segments']['train']['stop_row']:
            intervals = [('final', start, start + DAY)]
        else:
            intervals = []
            for fold in splits['oof_folds']:
                first = max(start, fold['predict_start'])
                last = min(start + DAY, fold['predict_stop'] + 300)
                if first < last:
                    intervals.append((fold['id'], first, last))
        for fold, first, last in intervals:
            frozen = FrozenCurveModel(output, model, fold, 2)
            for begin in range(first, last, config['inference_chunk_rows']):
                positions = np.arange(begin, min(last, begin + config['inference_chunk_rows']))
                forecast[positions - start] = frozen.predict_rows(matrix, positions)
        raw_alpha = forecast[:, 10].copy()
        if model != 'zero':
            calibration = json.loads((output / 'e1/calibration' / f'{model}.json').read_text())
            forecast = forecast * np.array(calibration['slope'], dtype=float) + np.array(calibration['intercept_bps'], dtype=float)
        timestamps = manifest['start_timestamp_ns'] + rows * SECOND
        table = pl.DataFrame({'timestamp_ns': timestamps, **{f'h{int(horizon)}': forecast[:, index]
                              for index, horizon in enumerate(HORIZONS)},
                              'edge_buy_bps': np.full(DAY, np.nan), 'edge_sell_bps': np.full(DAY, np.nan)})
        table.write_csv(destination)
        pl.DataFrame({'ready_ts_ns': timestamps + config['prediction_latency_ns'],
                      'raw_alpha_60s_bps': raw_alpha}).write_parquet(folder / f'{date}.raw.parquet')
    return destination


# 【非核心】从已下载官方月档形成当日资金费输入，分钟标记价仅作为明确的估值代理。
def make_funding_day(output: Path, config: dict, date: str) -> None:
    """带高低价区间，沿用已有独立资金费核算接口。"""
    destination = output / 'e4' / f'funding_{date}.json'
    if destination.exists():
        return
    manifest, splits, matrix, targets, market = open_dataset(output)
    day = manifest['dates'].index(date)
    funding_inputs(output, manifest, day * DAY, (day + 1) * DAY)
    cache = output / 'e3/funding'
    month = date[:7]
    with zipfile.ZipFile(cache / f'{month}_funding.zip') as archive:
        rates = pl.read_csv(archive.open(archive.namelist()[0]),
                            schema_overrides={'calc_time': pl.Int64, 'last_funding_rate': pl.Float64})
    marks = read_mark_prices(cache, month)
    start = int(np.datetime64(date, 'ms').astype(np.int64))
    rates = rates.filter(pl.col('calc_time').is_between(start, start + 86400000, closed='left'))
    events = []
    for record in rates.iter_rows(named=True):
        minute = record['calc_time'] // 60000 * 60000
        price = marks.filter(pl.col('open_time') == minute).row(0, named=True)
        events.append({'fundingTime': record['calc_time'], 'fundingRate': record['last_funding_rate'],
                       'markPrice': price['open'], 'markPriceLow': price['low'], 'markPriceHigh': price['high']})
    write_json_atomic(destination, {'records': events, 'valuation': 'official_minute_mark_open_proxy'})


# 【非核心】跨两个分析进程共享四个回放名额，避免扩展模型启动后并发翻倍。
@contextmanager
def replay_slot(output: Path):
    """文件锁只限制实际原生回放，信号生成、分析和下载可独立进行。"""
    folder = output / 'e4/.slots'
    folder.mkdir(parents=True, exist_ok=True)
    acquired = None
    while acquired is None:
        for index in range(4):
            descriptor = (folder / f'{index}.lock').open('a')
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = descriptor
                break
            except BlockingIOError:
                descriptor.close()
        if acquired is None:
            time.sleep(.2)
    try:
        yield
    finally:
        acquired.close()


# 【非核心】为一个真实行情日写出独立交易、委托和曲线配置，费用和单位逐项明确。
def case_configs(root: Path, config: dict, output: Path, case: dict, signal: Path) -> tuple:
    """风险方差来自Train OOF；独立机会回放实际不使用Alpha或风险去选择订单。"""
    date = case['date']
    start = int(np.datetime64(date, 'ns').astype(np.int64))
    folder = output / 'e4/configs'
    folder.mkdir(parents=True, exist_ok=True)
    settings = config['execution']
    model = case['model_id']
    covariance = np.eye(16)
    if case['mode'] not in ['fixed', 'opportunity']:
        calibration = json.loads((output / 'e1/calibration' / f'{model}.json').read_text())
        covariance = np.array(calibration['increment_error_covariance_bps2'])
    risks = np.maximum(np.diag(covariance), 1e-8)
    cumulative = np.maximum(covariance.cumsum(axis=0).cumsum(axis=1).diagonal(), 1e-8)
    risk_kind = 'identity' if case['mode'] in ['fixed', 'opportunity'] else 'calibrated'
    risk_file = folder / f'{model}.{risk_kind}.risk.txt'
    with file_locks([risk_file]):
        if not risk_file.exists():
            risk_file.write_text(' '.join(map(str, risks)) + '\n' + ' '.join(map(str, cumulative)) + '\n')
    source = output / 'l2/event_cache' / date / 'events/manifest.json'
    trading = {'name': case['id'], 'date': date, 'source': str(source),
               'output': str(output / 'e4/runs' / case['id']), 'tick_size': settings['tick_size'],
               'lot_size': settings['quantity_step_btc'], 'order_quantity': settings['risk_unit_btc'],
               'max_position': settings['risk_unit_btc'] * settings['max_position_units'],
               'initial_capital': settings['initial_equity_usdt'], 'maker_fee': settings['maker_fee'],
               'taker_fee': settings['taker_fee'], 'half_width_ticks': 1, 'decision_interval_ns': 100000000,
               'update_mode': 'time', 'price_threshold_ticks': 1,
               'entry_latency_ns': case['entry_latency_ns'], 'response_latency_ns': settings['response_latency_ns'],
               'quote_start_timestamp_ns': start + 120 * SECOND,
               'quote_stop_timestamp_ns': start + (DAY - 20) * SECOND}
    execution = {'mode': 'passive', 'queue_model': case['queue_model'], 'entry_bps': case['entry_bps'],
                 'exit_bps': .01, 'scale_bps': 1., 'max_alpha_bps': 5., 'inventory_skew_bps': 0.,
                 'cost_filter': 0, 'offset_ticks': case['offset_ticks'], 'ttl_ns': case['ttl_ms'] * 1000000,
                 'max_hold_ns': case['hold_s'] * SECOND, 'prediction_latency_ns': config['prediction_latency_ns'],
                 'taker_slippage_ticks': 1, 'quote_half_width_bps': 0.0,
                 'minimum_notional_usdt': settings['minimum_notional_usdt']}
    curve = {'mode': case['mode'], 'risk_lambda': case['risk_lambda'], 'cost_buffer_bps': case['buffer_bps'],
             'risk_unit_btc': settings['risk_unit_btc'], 'horizon_index': case['horizon_index'],
             'rebalance_interval_ns': case['rebalance_seconds'] * SECOND,
             'opportunity_interval_ns': 400 * SECOND, 'opportunity_side': case['side'],
             'edge_minimum_bps': 0., 'minimum_notional_usdt': settings['minimum_notional_usdt']}
    files = []
    configurations = [('trading', trading), ('execution', execution)]
    if case['mode'] != 'passive':
        configurations.append(('curve', curve))
    for name, values in configurations:
        file = folder / f'{case["id"]}.{name}.yaml'
        file.write_text(f'# E4 {name}配置，输入冻结模型与已验收事件，输出原生订单及账本。\n' + yaml.safe_dump(values, sort_keys=False))
        files.append(file)
    return files, risk_file, source


# 【核心逻辑285】实际C++逐事件成交后独立核算费用和资金费，失败保留原日志。
def execute_case(root: Path, config: dict, output: Path, case: dict, signal: Path) -> dict:
    """每个案例独占账本；完成产物原子发布，失败不会被解释为零收益。"""
    run = output / 'e4/runs' / case['id']
    metric_file = run / 'execution_metrics.json'
    failure_file = run / 'failed.json'
    with file_locks([output / 'e4/case_locks' / case['id']]):
        if metric_file.exists():
            return {'id': case['id'], 'status': 'complete'}
        if failure_file.exists():
            return {'id': case['id'], 'status': 'failed', 'failure': str(failure_file)}
        files, risk, source = case_configs(root, config, output, case, signal)
        binary = output / 'build/curve_replay'
        command = ['prlimit', '--as=4294967296', str(binary), *map(str, files), str(signal), str(risk)]
        if case['mode'] == 'passive':
            scalar = signal.with_name(signal.stem + '.scalar.csv')
            with file_locks([scalar]):
                if not scalar.exists():
                    pl.read_csv(signal).select('timestamp_ns', pl.col('h60').alias('prediction_bps')).write_csv(scalar)
            command = ['prlimit', '--as=4294967296', str(output / 'build/strategy_replay'),
                       str(files[0]), str(files[1]), str(scalar)]
        logs = output / 'e4/logs'
        logs.mkdir(exist_ok=True)
        with replay_slot(output), (logs / f'{case["id"]}.log').open('w') as log:
            completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            write_json_atomic(failure_file, {'case': case, 'exit_code': completed.returncode})
            return {'id': case['id'], 'status': 'failed', 'exit_code': completed.returncode}
        analysis_config = {'date': case['date'], 'source': str(source), 'output': str(output / 'e4'),
                           'maker_fee': config['execution']['maker_fee'], 'taker_fee': config['execution']['taker_fee'],
                           'initial_capital': config['execution']['initial_equity_usdt']}
        extra = run / 'curve_decisions.csv'
        if extra.exists():
            schema = {'local_timestamp_ns': pl.Int64, 'target_units': pl.Float64,
                      'actual_units': pl.Float64, 'reserved_buy': pl.Float64, 'reserved_sell': pl.Float64}
            pl.read_csv(extra, schema_overrides=schema).write_parquet(run / 'curve_decisions.parquet', compression='zstd')
            extra.unlink()
        metrics = analyze_strategy_case(root, analysis_config, case, run)
        metrics['execution_data'] = 'full_native_l2_and_official_trades'
        metrics['daily_account_reset'] = True
        metrics['edge_estimator_scope'] = 'fixed_exit_proxy' if case['mode'] == 'curve_edge' else 'not_used'
        write_json_atomic(metric_file, metrics)
    return {'id': case['id'], 'status': 'complete'}


# 【非核心】固定策略、队列和单参数情景，以显式ID保留每次原生回放。
def make_case(model: str, date: str, policy: str, mode: str, queue: str, **changes) -> dict:
    """默认规模0.002 BTC，机会回放的方向、TTL、距离与退出规则均进入唯一ID。"""
    case = {'model_id': model, 'date': date, 'policy_id': policy, 'mode': mode, 'queue_model': queue,
            'entry_latency_ns': 1000000, 'offset_ticks': 0, 'ttl_ms': 2000, 'hold_s': 300,
            'risk_lambda': 1., 'buffer_bps': 0., 'horizon_index': 10, 'rebalance_seconds': 1,
            'side': 1, 'entry_bps': .05}
    case.update(changes)
    case['id'] = (f'{model}__{date}__{policy}__{queue}__o{case["offset_ticks"]}__ttl{case["ttl_ms"]}'
                  f'__h{case["hold_s"]}__s{case["side"]}__lat{case["entry_latency_ns"]}'
                  f'__risk{case["risk_lambda"]}__b{case["buffer_bps"]}__r{case["rebalance_seconds"]}')
    return case


# 【核心逻辑286】固定订单样本按提交金额归一化收益，未成交和部分成交均保留。
def collect_opportunities(output: Path, case: dict) -> pl.DataFrame:
    """400秒间隔确保上一次300秒风险退出完成；分组Alpha按实际发单时已可用的原预测取得。"""
    run = output / 'e4/runs' / case['id']
    orders = pl.read_parquet(run / 'orders.parquet')
    fills = pl.read_parquet(run / 'fills.parquet')
    funding = pl.read_parquet(run / 'funding_cashflows.parquet')
    start = int(np.datetime64(case['date'], 'ns').astype(np.int64))
    interval = 400 * SECOND
    first = (start + 120 * SECOND + interval - 1) // interval * interval
    slots = np.arange(first, start + (DAY - 20) * SECOND, interval, dtype=np.int64)
    submissions = orders.filter((pl.col('kind') == 'submit') & (pl.col('order_id') < 1000000000000))
    records = []
    for slot in slots:
        submitted = submissions.filter(pl.col('local_timestamp_ns').is_between(slot, slot + interval, closed='left'))
        trades = fills.filter(pl.col('exchange_timestamp_ns').is_between(slot, slot + interval, closed='left'))
        entry = trades.filter(pl.col('maker') == 1)
        paid_funding = funding.filter(pl.col('timestamp_ns').is_between(slot, slot + interval, closed='left'))['funding_payment'].sum()
        quantity = float(entry['quantity'].sum())
        requested = float(submitted['quantity'][0]) if submitted.height else .002
        gross = -trades.select((pl.col('side') * pl.col('quantity') * pl.col('price')).sum()).item()
        net = gross - trades['fee'].sum() - paid_funding
        limit = float(submitted['price'][0]) if submitted.height else None
        submitted_time = int(submitted['local_timestamp_ns'][0]) if submitted.height else int(slot)
        records.append({'candidate_id': case['id'] + f'__{int(slot)}', 'date': case['date'],
                        'decision_ts_ns': submitted_time, 'model_id': case['model_id'], 'side': case['side'],
                        'offset_ticks': case['offset_ticks'], 'ttl_ms': case['ttl_ms'], 'hold_s': case['hold_s'],
                        'queue_model': case['queue_model'], 'submitted': submitted.height > 0,
                        'submitted_qty_btc': requested, 'filled_qty_btc': quantity,
                        'filled_any': quantity > 0, 'fully_filled': quantity >= requested - 1e-10,
                        'fill_fraction': quantity / requested, 'net_pnl_usdt': net,
                        'conditional_net_bps': net / (quantity * limit) * 10000 if quantity and limit else None,
                        'submitted_value_bps': net / (requested * limit) * 10000 if limit else 0.,
                        'exit_policy_id': f'fixed_hold_{case["hold_s"]}s_then_ioc',
                        'cycle_complete': bool(slot + (case['hold_s'] + 6) * SECOND < start + (DAY - 20) * SECOND)})
    table = pl.DataFrame(records)
    raw = pl.read_parquet(output / 'e4/signals' / case['model_id'] / f'{case["date"]}.raw.parquet')
    return table.sort('decision_ts_ns').join_asof(raw.sort('ready_ts_ns'),
        left_on='decision_ts_ns', right_on='ready_ts_ns', strategy='backward', tolerance=SECOND)


# 【非核心】只用固定退出的训练机会估计分组价值，样本不足时按预登记规则合并Alpha组。
def fit_fixed_exit_edges(output: Path, models: list[str], opportunities: pl.DataFrame) -> dict:
    """此估计明确属于固定60秒退出代理，不称为动态库存退出的精确EV。"""
    folder = output / 'e4/edge_models'
    folder.mkdir(exist_ok=True)
    estimates = {}
    dates = opportunities['date'].unique().sort().to_list()
    for model in models:
        existing = folder / f'{model}.json'
        if existing.exists():
            estimates[model] = json.loads(existing.read_text())
            continue
        records = []
        for date in dates:
            raw = pl.read_parquet(output / 'e4/signals' / model / f'{date}.raw.parquet')
            table = opportunities.filter(pl.col('date') == date).drop('raw_alpha_60s_bps', 'ready_ts_ns')
            records.append(table.sort('decision_ts_ns').join_asof(raw.sort('ready_ts_ns'),
                left_on='decision_ts_ns', right_on='ready_ts_ns', strategy='backward', tolerance=SECOND))
        table = pl.concat(records).filter(pl.col('submitted') & pl.col('cycle_complete')
                                           & (pl.col('hold_s') == 60) & pl.col('raw_alpha_60s_bps').is_finite())
        thresholds = table['raw_alpha_60s_bps'].quantile(.2), table['raw_alpha_60s_bps'].quantile(.4), table['raw_alpha_60s_bps'].quantile(.6), table['raw_alpha_60s_bps'].quantile(.8)
        groups = np.searchsorted(thresholds, table['raw_alpha_60s_bps'].to_numpy())
        table = table.with_columns(pl.Series('alpha_group', groups))
        cells = []
        for side in [-1, 1]:
            for queue in ['risk_adverse', 'log_probability']:
                for offset in [0, 1, 2, 5]:
                    for ttl in [500, 2000, 5000]:
                        pooled = table.filter((pl.col('side') == side) & (pl.col('queue_model') == queue)
                                              & (pl.col('offset_ticks') == offset) & (pl.col('ttl_ms') == ttl))
                        for group in range(5):
                            cell = pooled.filter(pl.col('alpha_group') == group)
                            merged = cell.height < 20
                            used = pooled if merged else cell
                            if not used.height:
                                continue
                            cells.append({'side': side, 'queue_model': queue, 'offset_ticks': offset,
                                          'ttl_ms': ttl, 'alpha_group': group, 'n_candidates': used.height,
                                          'merged_alpha_groups': merged, 'pred_fill_prob': used['filled_any'].mean(),
                                          'pred_fill_fraction': used['fill_fraction'].mean(),
                                          'pred_conditional_net_bps': used['conditional_net_bps'].mean(),
                                          'pred_submitted_value_bps': used['submitted_value_bps'].mean()})
        estimate = {'model_id': model, 'thresholds': list(thresholds), 'cells': cells,
                    'fit_split': 'train_oof', 'exit_policy_id': 'fixed_hold_60s_then_ioc',
                    'application_status': 'diagnostic_proxy_for_dynamic_inventory', 'source_dates': dates}
        write_json_atomic(folder / f'{model}.json', estimate)
        estimates[model] = estimate
    return estimates


# 【非核心】为两个方向分别输出当前可用的固定退出价值，阈值与格子均来自Train OOF。
def edge_signal(output: Path, model: str, date: str, queue: str, estimate: dict) -> Path:
    """保留原收益曲线，仅添加明确标记的执行价值代理列。"""
    folder = output / 'e4/signals' / model
    destination = folder / f'{date}.{queue}.edge_proxy.csv'
    if destination.exists():
        return destination
    table = pl.read_csv(folder / f'{date}.csv')
    raw = pl.read_parquet(folder / f'{date}.raw.parquet')['raw_alpha_60s_bps'].to_numpy()
    groups = np.searchsorted(estimate['thresholds'], raw)
    for side, name in [(1, 'edge_buy_bps'), (-1, 'edge_sell_bps')]:
        values = np.full(DAY, np.nan)
        for cell in estimate['cells']:
            if cell['side'] == side and cell['queue_model'] == queue and cell['offset_ticks'] == 0 and cell['ttl_ms'] == 2000:
                values[groups == cell['alpha_group']] = cell['pred_submitted_value_bps']
        table = table.with_columns(pl.Series(name, values))
    table.write_csv(destination)
    return destination


# 【非核心】汇总当前实际成功的事件回放与失败案例，完整账本仍按案例保留。
def execution_report(output: Path) -> None:
    """机会回放和主策略分别归档，单日重置账户与固定退出代理的限制明确记录。"""
    folder = output / 'e4'
    records = [json.loads(file.read_text()) for file in sorted((folder / 'runs').glob('*/execution_metrics.json'))]
    failures = [json.loads(file.read_text()) for file in sorted((folder / 'runs').glob('*/failed.json'))]
    if records:
        pl.DataFrame(records, infer_schema_length=None).write_csv(folder / 'strategy_metrics.csv')
    write_json_atomic(folder / 'execution_status.json', {'complete_cases': len(records), 'failed_cases': failures,
                      'native_days': sorted(set(record['date'] for record in records)),
                      'schema_version': 'multihorizon_execution_v1'})
    (folder / 'execution_report.md').write_text('# E4原生事件执行｜2026-11-09\n\n'
        'X0固定双边、X1单侧Alpha、X2单期限目标、X2完整曲线在实际L2和官方成交上回放。'
        '订单、排队、撤单在途、部分成交和费用均复用原生引擎，独立账本核对后才发布成功指标。'
        '每个日账户从相同资金和空仓开始，日末实际IOC平仓。'
        '独立机会以400秒间隔提交一次固定方向订单，保留未成交和部分成交，实际退出来自预先指定持有时间。'
        '执行分组价值仅由Train OOF机会拟合。X3_proxy将固定60秒退出价值作为动态库存的筛选诊断，'
        '尚未将它称为动态退出策略的精确完整交易EV，因此计划严格X3/E5门槛仍需补齐。\n')


# 【非核心】固定订单机会仅按预登记日期、方向和委托参数展开，不依赖库存策略选参。
def training_opportunity_cases(output: Path) -> list[dict]:
    """返回前六个保留训练日的384个案例，供提前回放与正式执行阶段共同使用。"""
    plan = json.loads((output / 'l2/batch_plan.json').read_text())
    retained = plan['retain_event_cache_dates']
    train_dates = [date for date in retained if '2025-10-01' <= date <= '2026-03-22']
    train_dates = train_dates[:6]
    cases = []
    for date in train_dates:
        for queue in ['risk_adverse', 'log_probability']:
            for side in [-1, 1]:
                for offset in [0, 1, 2, 5]:
                    for ttl in [500, 2000, 5000]:
                        cases.append(make_case('M1', date, 'OPP', 'opportunity', queue,
                                               side=side, offset_ticks=offset, ttl_ms=ttl, hold_s=60))
                for hold in [10, 30, 120, 300]:
                    cases.append(make_case('M1', date, 'OPP', 'opportunity', queue, side=side, hold_s=hold))
    return cases


# 【核心逻辑288】固定机会独立于E2和E3提前执行，复用同一案例锁及四个原生回放名额。
def run_training_opportunities(filename: Path, baseline_only: bool = False) -> None:
    """固定双边基线与训练机会提前回放，正式E4复用完成账本，截止前十分钟停止新增案例。"""
    root, config, output = context(filename)
    opportunities = training_opportunity_cases(output)
    write_json_atomic(output / 'e4/opportunity_case_plan.json', opportunities)
    retained = json.loads((output / 'l2/batch_plan.json').read_text())['retain_event_cache_dates']
    validation_dates = [date for date in retained if '2026-03-23' <= date <= '2026-07-06']
    baselines = [make_case('zero', date, 'X0_fixed', 'fixed', queue)
                 for date in validation_dates for queue in ['risk_adverse', 'log_probability']]
    cases = baselines if baseline_only else opportunities
    name = 'early_baselines' if baseline_only else 'early_opportunities'
    write_json_atomic(output / 'e4' / f'{name}_case_plan.json', cases)
    deadline = dt.datetime.fromisoformat(config['stop_at']).timestamp() - 600
    pending = list(cases)
    while pending and time.time() < deadline:
        ready = []
        for case in pending:
            validation_file = output / 'l2/validation' / f'{case["date"]}.json'
            if validation_file.exists():
                validation = json.loads(validation_file.read_text())
                if validation['native_replay_validation'] == 'passed':
                    ready.append(case)
        if not ready:
            time.sleep(30)
            continue
        date = ready[0]['date']
        model = ready[0]['model_id']
        signal = export_day(output, config, model, date)
        make_funding_day(output, config, date)
        batch = [case for case in ready if case['date'] == date and case['model_id'] == model][:4]
        with ThreadPoolExecutor(max_workers=4) as workers:
            futures = {workers.submit(execute_case, root, config, output, case, signal): case for case in batch}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    result = future.result()
                except (ValueError, OSError, RuntimeError, pl.exceptions.PolarsError) as error:
                    failure = output / 'e4/runs' / case['id'] / 'failed.json'
                    write_json_atomic(failure, {'case': case, 'stage': 'execution_or_ledger', 'error': str(error)})
                    result = {'id': case['id'], 'status': 'failed', 'failure': str(failure)}
                print('EARLY_EXECUTION', result, flush=True)
        completed_ids = {case['id'] for case in batch}
        pending = [case for case in pending if case['id'] not in completed_ids]
        write_json_atomic(output / 'e4' / f'{name}_progress.json',
                          {'planned': len(cases), 'handled': len(cases) - len(pending)})


# 【核心逻辑287】逐日消费已验收L2，主策略和独立机会并行，只有训练机会完整后才拟合代理过滤。
def run_execution(filename: Path) -> None:
    """固定矩阵逐步执行至07:50，后续08:00由外部监督保存结果并停止本任务树。"""
    root, config, output = context(filename)
    folder = output / 'e4'
    folder.mkdir(exist_ok=True)
    manifest, splits, matrix, targets, market = open_dataset(output)
    selected = json.loads((output / 'e1/selected_models.json').read_text())['models']
    registry = json.loads((output / 'e3/policy_registry.json').read_text())
    choices = json.loads((output / 'e3/parameter_selection.json').read_text())['models']
    retained = json.loads((output / 'l2/batch_plan.json').read_text())['retain_event_cache_dates']
    cases = training_opportunity_cases(output)
    train_dates = sorted({case['date'] for case in cases})
    validation_dates = [date for date in retained if '2026-03-23' <= date <= '2026-07-06']
    for model in selected:
        calibration = json.loads((output / 'e1/calibration' / f'{model}.json').read_text())
        for date in validation_dates:
            for queue in ['risk_adverse', 'log_probability']:
                if model == 'M1':
                    cases.append(make_case('zero', date, 'X0_fixed', 'fixed', queue))
                threshold = abs(calibration['quantile_thresholds'][10][-1])
                cases.append(make_case(model, date, 'X1_passive', 'passive', queue,
                                       entry_bps=max(.05, threshold), hold_s=60))
                for policy, mode in [('P2', 'single'), ('P3', 'curve')]:
                    selected_case = next(case for case in registry[model] if case['id'] == choices[model][policy])
                    cases.append(make_case(model, date, f'X2_{mode}', mode, queue,
                        risk_lambda=selected_case['risk_lambda'], buffer_bps=selected_case['buffer_bps'],
                        horizon_index=selected_case['horizon_index'], rebalance_seconds=selected_case['rebalance_seconds'],
                        hold_s=selected_case['max_hold_seconds']))
    plan_file = folder / f'case_plan_{"_".join(selected)}.json'
    write_json_atomic(plan_file, cases)
    deadline = dt.datetime.fromisoformat(config['stop_at']).timestamp() - 600
    results = []
    handled = set()
    edge_models = None
    while time.time() < deadline:
        available = {}
        for file in (output / 'l2/validation').glob('*.json'):
            record = json.loads(file.read_text())
            if record['native_replay_validation'] == 'passed':
                available[record['date']] = record
        ready = [case for case in cases if case['id'] not in handled and case['date'] in available]
        if ready:
            signals = {}
            base_signals = {}
            for case in ready:
                key = case['model_id'], case['date']
                if key not in base_signals:
                    base_signals[key] = export_day(output, config, *key)
                    make_funding_day(output, config, case['date'])
                signals[case['id']] = base_signals[key]
                if case['mode'] == 'curve_edge':
                    signals[case['id']] = edge_signal(output, *key, case['queue_model'], edge_models[case['model_id']])
            with ThreadPoolExecutor(max_workers=4) as workers:
                futures = {workers.submit(execute_case, root, config, output, case,
                            signals[case['id']]): case for case in ready}
                for future in as_completed(futures):
                    case = futures[future]
                    try:
                        result = future.result()
                    except (ValueError, OSError, RuntimeError, pl.exceptions.PolarsError) as error:
                        failure = folder / 'runs' / case['id'] / 'failed.json'
                        write_json_atomic(failure, {'case': case, 'stage': 'execution_or_ledger', 'error': str(error)})
                        result = {'id': case['id'], 'status': 'failed', 'failure': str(failure)}
                    results.append(result)
                    handled.add(case['id'])
                    write_json_atomic(folder / f'progress_{"_".join(selected)}.json', {'results': results, 'total_cases': len(cases)})
                    print('E4_CASE', result['status'], case['id'], flush=True)
        opportunity_cases = [case for case in cases if case['policy_id'] == 'OPP']
        opportunities_done = all((folder / 'runs' / case['id'] / 'execution_metrics.json').exists() for case in opportunity_cases)
        if opportunities_done and edge_models is None:
            opportunity_file = folder / 'opportunities.parquet'
            with file_locks([opportunity_file]):
                if opportunity_file.exists():
                    opportunities = pl.read_parquet(opportunity_file)
                else:
                    opportunities = pl.concat([collect_opportunities(output, case) for case in opportunity_cases])
                    opportunities.write_parquet(opportunity_file, compression='zstd')
            for model in selected:
                for date in train_dates:
                    export_day(output, config, model, date)
            edge_models = fit_fixed_exit_edges(output, selected, opportunities)
            additional = []
            for case in cases:
                if case['mode'] == 'curve' and case['policy_id'] == 'X2_curve':
                    proxy = dict(case)
                    proxy.update(policy_id='X3_fixed_exit_proxy', mode='curve_edge', id=case['id'].replace('X2_curve', 'X3_fixed_exit_proxy'))
                    additional.append(proxy)
            cases.extend(additional)
            write_json_atomic(plan_file, cases)
        registered = json.loads((output / 'e1/model_registry.json').read_text())
        ready_models = {record['model_id'] for record in registered}
        current_selection = json.loads((output / 'e1/selected_models.json').read_text())['models']
        if edge_models is not None and {'M1', 'M2', 'M3', 'M4'} <= ready_models and current_selection == selected:
            for date in train_dates + validation_dates:
                date_cases = [case for case in cases if case['date'] == date]
                complete = all((folder / 'runs' / case['id'] / 'execution_metrics.json').exists() for case in date_cases)
                if complete:
                    write_json_atomic(output / 'l2/consumed' / f'{date}.json', {
                        'case_plan': str(plan_file), 'completed_cases': len(date_cases),
                        'selected_models': selected, 'status': 'all_frozen_execution_cases_complete',
                    })
        if len(handled) == len(cases):
            break
        execution_report(output)
        time.sleep(30)
    execution_report(output)
    write_json_atomic(folder / f'completed_{"_".join(selected)}.json', {'handled': len(handled), 'planned': len(cases),
                      'unfinished_cases': [case['id'] for case in cases if case['id'] not in handled]})
