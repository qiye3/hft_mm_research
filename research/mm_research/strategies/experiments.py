"""固定研究矩阵的真实事件回放队列，失败独立保存且不覆盖成功产物。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import polars as pl
import yaml

from mm_research.data.storage import write_json_atomic


# 【核心逻辑234】在收益回看前固定模型、四类策略和单参数敏感性，不按回看收益挑选参数。
def make_strategy_cases(config: dict, signals: list[dict], output: Path) -> list[dict]:
    """普通费率主实验配两种排队，另存成本预算关闭的机制诊断。"""
    day = int(np.datetime64(config['date'], 'ns').astype(np.int64))
    zero_file = output / 'signals/zero.csv'
    pl.DataFrame({'timestamp_ns': day + np.arange(86400, dtype=np.int64) * 1_000_000_000,
                  'prediction_bps': np.zeros(86400)}).write_csv(zero_file)
    cases = []
    for queue in config['queues']:
        cases.append({'signal_id': 'zero', 'signal_file': str(zero_file), 'mode': 'fixed', 'queue_model': queue,
                      'entry_bps': 0., 'exit_bps': 0., 'scale_bps': 1., 'hold_s': 60,
                      'offset_ticks': 0, 'ttl_ms': config['order_ttl_ms'], 'cost_filter': False,
                      'category': 'S0'})
    for signal in signals:
        if signal['train_abs_q90_bps'] <= 0:
            raise ValueError(f'常数信号不具备方向性交易定义 {signal["id"]}')
        entry = max(config['minimum_entry_bps'], signal['train_abs_q80_bps'])
        base = {'signal_id': signal['id'], 'signal_file': signal['file'], 'entry_bps': entry,
                'exit_bps': min(entry, signal['train_abs_q50_bps']), 'scale_bps': signal['train_abs_q90_bps'],
                'hold_s': min(config['maximum_hold_seconds'], max(config['minimum_hold_seconds'], signal['horizon_s'])),
                'offset_ticks': config['offset_ticks'], 'ttl_ms': config['order_ttl_ms']}
        for queue in config['queues']:
            for mode in config['modes']:
                cases.append({**base, 'mode': mode, 'queue_model': queue, 'cost_filter': True, 'category': 'main'})
        if config['mechanism_diagnostic']:
            cases.append({**base, 'mode': 'passive', 'queue_model': 'risk_adverse', 'cost_filter': False,
                          'category': 'mechanism'})
        if signal['id'] in config['sensitivity_signals']:
            sensitivity = [(offset, base['ttl_ms'], base['hold_s']) for offset in config['offset_sensitivity']]
            sensitivity += [(base['offset_ticks'], ttl, base['hold_s']) for ttl in config['ttl_sensitivity_ms']]
            sensitivity += [(base['offset_ticks'], base['ttl_ms'], hold) for hold in config['hold_sensitivity_seconds']]
            for offset, ttl, hold in sensitivity:
                cases.append({**base, 'mode': 'passive', 'queue_model': 'risk_adverse', 'cost_filter': False,
                              'offset_ticks': offset, 'ttl_ms': ttl, 'hold_s': hold, 'category': 'S2_sensitivity'})
    unique = {}
    for case in cases:
        case['quote_half_width_bps'] = 0.0
        case['inventory_skew_bps'] = 0.0
        case['minimum_notional_usdt'] = 100.0
        identifier = (f'{case["signal_id"]}__{case["mode"]}__{case["queue_model"]}'
                      f'__o{case["offset_ticks"]}__ttl{case["ttl_ms"]}__h{case["hold_s"]}__c{int(case["cost_filter"])}')
        if identifier not in unique:
            unique[identifier] = {'id': identifier, **case}
    return list(unique.values())


# 【核心逻辑235】分别生成交易参数与策略规则配置，所属阶段、输入和输出在文件开头说明。
def write_case_configs(root: Path, config: dict, case: dict, output: Path) -> tuple[Path, Path]:
    """交易配置沿用M1公开入口，策略参数仅负责预测如何变成订单。"""
    day = int(np.datetime64(config['date'], 'ns').astype(np.int64))
    folder = output / 'configs'
    folder.mkdir(exist_ok=True)
    trading = {
        'name': case['id'], 'date': config['date'], 'source': str(root / config['source']),
        'output': str(output / 'runs' / case['id']), 'tick_size': .1, 'lot_size': .001,
        'order_quantity': config['order_quantity'], 'max_position': config['max_position'],
        'initial_capital': config['initial_capital'], 'maker_fee': config['maker_fee'], 'taker_fee': config['taker_fee'],
        'half_width_ticks': 1, 'decision_interval_ns': config['decision_interval_ns'], 'update_mode': 'time',
        'price_threshold_ticks': 1, 'entry_latency_ns': config['entry_latency_ns'],
        'response_latency_ns': config['response_latency_ns'],
        'quote_start_timestamp_ns': day + config['warmup_seconds'] * 1_000_000_000,
        'quote_stop_timestamp_ns': day + (86400 - config['liquidation_buffer_seconds']) * 1_000_000_000,
    }
    policy = {
        'mode': case['mode'], 'queue_model': case['queue_model'], 'entry_bps': case['entry_bps'],
        'exit_bps': case['exit_bps'], 'scale_bps': case['scale_bps'], 'max_alpha_bps': config['max_alpha_bps'],
        'inventory_skew_bps': case['inventory_skew_bps'], 'cost_filter': int(case['cost_filter']), 'offset_ticks': case['offset_ticks'],
        'ttl_ns': case['ttl_ms'] * 1_000_000, 'max_hold_ns': case['hold_s'] * 1_000_000_000,
        'quote_half_width_bps': case['quote_half_width_bps'],
        'minimum_notional_usdt': case['minimum_notional_usdt'],
        'prediction_latency_ns': config['prediction_latency_ns'], 'taker_slippage_ticks': config['taker_slippage_ticks'],
    }
    trading_file = folder / f'{case["id"]}.trading.yaml'
    policy_file = folder / f'{case["id"]}.policy.yaml'
    trading_file.write_text('# Trading配置，输入原生NPZ，输出订单、成交和权益账本。\n' + yaml.safe_dump(trading, sort_keys=False))
    policy_file.write_text('# Strategies配置，输入bps预测和原生状态，输出目标挂单与IOC退出。\n' + yaml.safe_dump(policy, sort_keys=False))
    return trading_file, policy_file


# 【核心逻辑244】独立进程执行一个策略，使用Linux地址空间上限控制单次原生回放的内存。
def run_strategy_case(root: Path, config: dict, case: dict, output: Path) -> dict:
    """失败保留日志，成功后核对逐笔现金流，再将CSV压缩为正式Parquet产物。"""
    from mm_research.strategies.analysis import analyze_strategy_case
    run = output / 'runs' / case['id']
    result_file = run / 'execution_metrics.json'
    if result_file.exists():
        for name in ['orders', 'fills', 'equity', 'liquidation_equity', 'ioc_actions']:
            source = run / f'{name}.csv'
            if source.exists():
                source.unlink()
        return {'id': case['id'], 'status': 'complete', 'metrics': json.loads(result_file.read_text())}
    if run.exists():
        raise RuntimeError(f'失败产物已存在，需要先定位失败原因并使用新的运行版本 {run}')
    if config['skip_proven_no_entry'] and case['cost_filter'] and case['mode'] in ['passive', 'inventory', 'hybrid']:
        maximum_alpha = pl.read_csv(case['signal_file'])['prediction_bps'].abs().max()
        minimum_gate = max(case['entry_bps'], 2 * config['maker_fee'] * 10000)
        if maximum_alpha < minimum_gate:
            run.mkdir(parents=True)
            proof = {**case, 'date': config['date'], 'result_origin': 'no_order_proof',
                     'native_replay_executed': False, 'date_max_abs_alpha_bps': maximum_alpha,
                     'minimum_entry_gate_bps': minimum_gate, 'fills': 0, 'maker_fills': 0, 'taker_fills': 0,
                     'turnover_usdt': 0., 'gross_trading_pnl_usdt': 0., 'trading_fees_usdt': 0.,
                     'net_trading_pnl_usdt': 0., 'funding_payment_usdt': 0., 'net_pnl_after_funding_usdt': 0.,
                     'funding_status': 'zero_position_all_day', 'max_abs_position_btc': 0., 'final_position_btc': 0.,
                     'proof': 'abs(alpha) never reaches entry gate; flat initial state implies no orders or funding exposure'}
            write_json_atomic(result_file, proof)
            return {'id': case['id'], 'status': 'complete', 'metrics': proof}
    trading, policy = write_case_configs(root, config, case, output)
    log_file = output / 'logs' / f'{case["id"]}.log'
    log_file.parent.mkdir(exist_ok=True)
    command = ['prlimit', '--as=4294967296', str(root / config['binary']), str(trading), str(policy), case['signal_file']]
    with log_file.open('w') as log:
        completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        return {'id': case['id'], 'status': 'failed', 'returncode': completed.returncode, 'log': str(log_file)}
    try:
        metrics = analyze_strategy_case(root, config, case, run)
    except (ValueError, OSError, pl.exceptions.PolarsError) as error:
        return {'id': case['id'], 'status': 'failed', 'stage': 'ledger_analysis', 'error': str(error)}
    return {'id': case['id'], 'status': 'complete', 'metrics': metrics}


# 【核心逻辑238】冻结实验矩阵后以有限并发运行，逐个保存成功和失败状态供持续跟进。
def run_strategy_sweep(root: Path, config: dict) -> None:
    """并发数由配置控制，任何失败不会转换为零收益或阻止其他模型。"""
    output = root / config['output']
    signals = json.loads((output / 'signal_exports.json').read_text())
    pool = json.loads((output / 'signal_pool_frozen.json').read_text())
    if len(signals) != len(pool):
        raise RuntimeError('完整预测尚未生成，必须先完成signals阶段')
    case_file = output / 'strategy_cases.json'
    if case_file.exists():
        cases = json.loads(case_file.read_text())
    else:
        cases = make_strategy_cases(config, signals, output)
        write_json_atomic(case_file, cases)
        write_json_atomic(output / 'research_config.json', config)
    results = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=config['max_parallel_replays']) as pool:
        futures = {pool.submit(run_strategy_case, root, config, case, output): case['id'] for case in cases}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_json_atomic(output / 'strategy_progress.json', {'total': len(cases), 'finished': len(results),
                'complete': sum(record['status'] == 'complete' for record in results),
                'failed': sum(record['status'] == 'failed' for record in results),
                'elapsed_seconds': time.time() - started, 'results': results})
            print('STRATEGY_CASE', result['status'], result['id'], flush=True)
    write_json_atomic(output / 'strategy_completed.json', {'total': len(cases), 'results': results})
