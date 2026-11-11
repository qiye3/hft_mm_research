"""冻结信号的逐日原生做市实验，按开发期选参并保存完整费用后账本。"""

import json
from pathlib import Path
import shutil

import numpy as np
import polars as pl

from mm_research.data.storage import file_locks, write_json_atomic
from mm_research.multihorizon.common import DAY, HORIZONS, SECOND
from mm_research.multihorizon.execution import (
    collect_opportunities, execute_case, make_case, make_funding_day,
)
from mm_research.strategies.analysis import analyze_strategy_case
from mm_research.strategies.experiments import write_case_configs
from mm_research.strategies.forecast_research import research_context


# 【非核心】复用官方资金费及已有L2路径，只在本轮目录保存所需的轻量来源。
def prepare_execution_day(config: dict, source: Path, output: Path, date: str) -> Path:
    """资金费沿用独立现金流核算，零信号只用于固定报价和固定订单机会。"""
    (output / 'e4').mkdir(exist_ok=True)
    funding = output / f'funding_{date}.json'
    with file_locks([funding]):
        if not funding.exists():
            make_funding_day(source, config, date)
            temporary = funding.with_suffix('.partial.json')
            shutil.copyfile(source / 'e4' / funding.name, temporary)
            temporary.replace(funding)
        probe_funding = output / 'e4' / funding.name
        if not probe_funding.exists():
            temporary = probe_funding.with_suffix('.partial.json')
            shutil.copyfile(funding, temporary)
            temporary.replace(probe_funding)
    signals = output / 'signals'
    signals.mkdir(exist_ok=True)
    zero = signals / f'zero_{date}.csv'
    with file_locks([zero]):
        if not zero.exists():
            start = int(np.datetime64(date, 'ns').astype(np.int64))
            pl.DataFrame({'timestamp_ns': start + np.arange(DAY) * SECOND,
                          'prediction_bps': np.zeros(DAY)}).write_csv(zero)
    return zero


# 【核心逻辑324】固定发单时刻和方向，记录真实成交及完整退出，Alpha只用于事后分组。
def run_probe_day(filename: Path, date: str) -> None:
    """每400秒一个独立机会，比较两侧、两种距离和TTL，使用保守排队。"""
    config, source, output = research_context(filename)
    prepare_execution_day(config, source, output, date)
    folder = output / 'probes'
    folder.mkdir(exist_ok=True)
    zero = output / 'signals' / f'zero_curve_{date}.csv'
    if not zero.exists():
        timestamps = int(np.datetime64(date, 'ns').astype(np.int64)) + np.arange(DAY) * SECOND
        pl.DataFrame({'timestamp_ns': timestamps,
                      **{f'h{horizon}': np.zeros(DAY) for horizon in HORIZONS},
                      'edge_buy_bps': np.full(DAY, np.nan),
                      'edge_sell_bps': np.full(DAY, np.nan)}).write_csv(zero)
    raw_folder = output / 'e4/signals/zero'
    raw_folder.mkdir(parents=True, exist_ok=True)
    timestamps = int(np.datetime64(date, 'ns').astype(np.int64)) + np.arange(DAY) * SECOND
    pl.DataFrame({'ready_ts_ns': timestamps + config['prediction_latency_ns'],
                  'raw_alpha_60s_bps': np.zeros(DAY)}).write_parquet(raw_folder / f'{date}.raw.parquet')
    for side in [1, -1]:
        for offset, ttl in [(0, 1000), (0, 3000), (5, 1000)]:
            case = make_case('zero', date, 'fixed_opportunity', 'opportunity', 'risk_adverse',
                             side=side, hold_s=180, offset_ticks=offset, ttl_ms=ttl)
            status = execute_case(Path.cwd(), config, output, case, zero)
            if status['status'] != 'complete':
                raise RuntimeError(f'固定订单机会回放失败 {case["id"]}')
            table = collect_opportunities(output, case)
            fills = pl.read_parquet(output / 'e4/runs' / case['id'] / 'fills.parquet').filter(pl.col('maker') == 1)
            marks = [f'markout_{suffix}_bps' for suffix in ['100ms', '1s', '5s', '10s', '30s', '60s', '120s', '180s', '300s']]
            matched = fills.sort('exchange_timestamp_ns').join_asof(
                table.select('decision_ts_ns').sort('decision_ts_ns'),
                left_on='exchange_timestamp_ns', right_on='decision_ts_ns', strategy='backward', tolerance=400 * SECOND,
            ).group_by('decision_ts_ns').agg(*[
                ((pl.col(name) * pl.col('quantity')).sum() /
                 pl.col('quantity').filter(pl.col(name).is_not_null()).sum()).alias(name) for name in marks
            ])
            table = table.join(matched, on='decision_ts_ns', how='left')
            table.write_parquet(folder / f'{case["id"]}.parquet', compression='zstd')
    write_json_atomic(folder / f'{date}.completed.json', {'status': 'success'})


# 【非核心】声明固定粗网格，每个变体只改变等待时间、报价宽度或持仓时长之一。
def strategy_variants(config: dict, *, pilot: bool) -> list[dict]:
    """六组固定信号分别进入S1/S2/S3，S0为统一风险预算的固定双边基线。"""
    cases = [{'signal_id': 'zero', 'mode': 'fixed', 'policy': 'S0', 'width_bps': width,
              'ttl_ms': 1000, 'hold_s': 180, 'dual': False} for width in ([0.] if pilot else [0., 2., 4.])]
    for signal in config['signals']:
        for mode, policy in [('passive', 'S1'), ('skew', 'S2'), ('inventory', 'S3')]:
            variants = [(0., 1000, 180)]
            if not pilot:
                if mode == 'passive':
                    variants.append((0., 3000, 180))
                elif mode == 'skew':
                    variants.extend([(2., 1000, 180), (4., 1000, 180)])
                else:
                    variants.extend([(0., 1000, 60), (0., 1000, 300)])
            for width, ttl, hold in variants:
                cases.append({'signal_id': signal['id'], 'mode': mode, 'policy': policy,
                              'width_bps': width, 'ttl_ms': ttl, 'hold_s': hold, 'dual': False})
        if not pilot and signal['horizon_s'] == 180:
            cases.append({'signal_id': signal['id'], 'mode': 'inventory', 'policy': 'S3',
                          'width_bps': 0., 'ttl_ms': 1000, 'hold_s': 180, 'dual': True})
    return cases


# 【非核心】将固定规则转换为唯一实验名，重启时能直接定位已完成案例。
def variant_name(variant: dict) -> str:
    """名称包含所有变动参数，费用和风险预算由冻结主配置提供。"""
    return (f'{variant["signal_id"]}_{variant["policy"]}_w{variant["width_bps"]}'
            f'_ttl{variant["ttl_ms"]}_h{variant["hold_s"]}_dual{int(variant["dual"])}')


# 【核心逻辑327】将长周期库存信号与短周期方向按同一已知时刻关联，冲突时暂停新增目标。
def execution_signal(output: Path, variant: dict, date: str) -> Path:
    """缺失或冲突置NaN，订单风险退出继续由原生策略处理，目标标签不参与信号生成。"""
    if variant['signal_id'] == 'zero':
        return output / 'signals' / f'zero_{date}.csv'
    signal = output / 'signals' / f'{variant["signal_id"]}_{date}.csv'
    if not variant['dual']:
        return signal
    destination = output / 'signals' / f'{variant["signal_id"]}_dual_{date}.csv'
    if not destination.exists():
        long = pl.read_csv(signal)
        short = pl.read_csv(output / 'signals' / f'h30_old_mlp_{date}.csv').rename({'prediction_bps': 'short_bps'})
        long.join(short, on='timestamp_ns', validate='1:1').select('timestamp_ns',
            pl.when(pl.col('prediction_bps') * pl.col('short_bps') >= 0)
            .then(pl.col('prediction_bps')).otherwise(float('nan')).alias('prediction_bps')).write_csv(destination)
    return destination


# 【核心逻辑326】实际事件回放后核对账本，恢复只复用成功案例，失败尝试保留独立目录。
def run_variant(config: dict, source: Path, output: Path, date: str, variant: dict,
                queue: str, *, latency_ns: int = 1_000_000, quantity: float = .002) -> dict:
    """重试最多三次，独立成功标记最后发布，失败不会被解释为零收益。"""
    import subprocess

    identifier = f'{date}_{variant_name(variant)}_{queue}_lat{latency_ns}_qty{quantity}'
    journal = output / 'cases'
    journal.mkdir(exist_ok=True)
    marker = journal / f'{identifier}.json'
    with file_locks([marker]):
        if marker.exists():
            return json.loads(marker.read_text())
        if variant['signal_id'] == 'zero':
            calibration = {'entry_bps': 0., 'exit_bps': 0., 'scale_bps': 1.}
        else:
            calibrations = json.loads((output / 'calibration.json').read_text())
            calibration = calibrations[variant['signal_id']]
        trading_config = {
            'date': date, 'source': str(source / 'l2/event_cache' / date / 'events/manifest.json'),
            'output': str(output), 'initial_capital': config['execution']['initial_equity_usdt'],
            'order_quantity': quantity, 'max_position': quantity * config['execution']['max_position_units'],
            'maker_fee': config['execution']['maker_fee'], 'taker_fee': config['execution']['taker_fee'],
            'entry_latency_ns': latency_ns, 'response_latency_ns': config['execution']['response_latency_ns'],
            'decision_interval_ns': 100_000_000, 'prediction_latency_ns': config['prediction_latency_ns'],
            'warmup_seconds': 120, 'liquidation_buffer_seconds': 20, 'max_alpha_bps': 5., 'taker_slippage_ticks': 1,
        }
        for attempt in range(1, 4):
            case = {
                **variant, 'id': f'{identifier}_attempt{attempt}', 'date': date,
                'mode': variant['mode'], 'queue_model': queue,
                'signal_file': str(execution_signal(output, variant, date).resolve()),
                'entry_bps': calibration['entry_bps'], 'exit_bps': calibration['exit_bps'],
                'scale_bps': max(calibration['scale_bps'], 1e-6), 'hold_s': variant['hold_s'],
                'offset_ticks': 0, 'cost_filter': False, 'category': variant['policy'],
                'quote_half_width_bps': variant['width_bps'],
                'inventory_skew_bps': .5 if variant['mode'] == 'skew' else 0.,
                'minimum_notional_usdt': config['execution']['minimum_notional_usdt'],
            }
            run = output / 'runs' / case['id']
            metrics_file = run / 'execution_metrics.json'
            if metrics_file.exists():
                metrics = json.loads(metrics_file.read_text())
            elif run.exists():
                # 已失败或中断的尝试保留原样，继续下一个独立尝试。
                continue
            else:
                trading_file, policy_file = write_case_configs(Path.cwd(), trading_config, case, output)
                logs = output / 'logs'
                logs.mkdir(exist_ok=True)
                with (logs / f'{case["id"]}.log').open('w') as log:
                    completed = subprocess.run(['prlimit', '--as=4294967296', str(output / 'build/strategy_replay'),
                        str(trading_file), str(policy_file), case['signal_file']], stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode != 0:
                    write_json_atomic(run / 'failed.json', {'returncode': completed.returncode, 'attempt': attempt})
                    continue
                metrics = analyze_strategy_case(Path.cwd(), trading_config, case, run)
            metrics.update(variant_id=variant_name(variant), date=date, queue_model=queue,
                           latency_ns=latency_ns, quantity_btc=quantity, daily_account_reset=True)
            write_json_atomic(marker, metrics)
            return metrics
        raise RuntimeError(f'案例三次尝试均未完成 {identifier}')


# 【核心逻辑325】开发期统一比较粗网格，冻结验证及压力阶段只读取已选定配置。
def run_day_stage(filename: Path, stage: str, date: str) -> None:
    """每个日期由独立队列任务执行，阶段之间通过冻结选择清单连接。"""
    config, source, output = research_context(filename)
    prepare_execution_day(config, source, output, date)
    if stage == 'baseline':
        if date in config['dates']['val_dev']:
            variants = [variant for variant in strategy_variants(config, pilot=False) if variant['policy'] == 'S0']
            queues = ['risk_adverse']
        else:
            variants = strategy_variants(config, pilot=True)[:1]
            queues = ['risk_adverse', 'log_probability']
    elif stage in ['pilot', 'develop']:
        variants = strategy_variants(config, pilot=stage == 'pilot')
        queues = ['risk_adverse']
    else:
        variants = json.loads((output / 'selected_strategies.json').read_text())['variants']
        variants = [*strategy_variants(config, pilot=True)[:1], *variants]
        queues = ['risk_adverse', 'log_probability']
    failures = []
    for variant in variants:
        for queue in queues:
            changes = [(1_000_000, .002)] if stage != 'stress' else [(5_000_000, .002), (20_000_000, .002), (1_000_000, .004)]
            for latency, quantity in changes:
                try:
                    run_variant(config, source, output, date, variant, queue,
                                latency_ns=latency, quantity=quantity)
                except (RuntimeError, ValueError, OSError, pl.exceptions.PolarsError) as error:
                    failures.append(f'{variant_name(variant)} {queue} {error}')
    write_json_atomic(output / f'{stage}_{date}.json', {'status': 'failed' if failures else 'success', 'failures': failures})
    if failures:
        raise RuntimeError(f'{date}存在失败案例，其他案例已继续执行')


# 【核心逻辑328】仅按开发期共同12日费用后收益选择每类策略的一个固定方案。
def freeze_strategies(filename: Path) -> None:
    """缺少任一开发日的候选不参与选择，验证期收益不读取。"""
    config, source, output = research_context(filename)
    records = [json.loads(path.read_text()) for path in (output / 'cases').glob('*.json')]
    table = pl.DataFrame(records, infer_schema_length=None).filter(
        pl.col('date').is_in(config['dates']['val_dev']) & (pl.col('queue_model') == 'risk_adverse')
        & (pl.col('latency_ns') == 1_000_000) & (pl.col('quantity_btc') == .002))
    scores = table.group_by('policy', 'variant_id').agg(
        pl.col('date').n_unique().alias('days'), pl.col('net_pnl_after_funding_usdt').sum().alias('net_pnl'),
        pl.col('net_pnl_after_funding_usdt').is_not_null().all().alias('funding_complete'),
    ).filter((pl.col('days') == len(config['dates']['val_dev'])) & pl.col('funding_complete'))
    selected = []
    available = {variant_name(item): item for item in strategy_variants(config, pilot=False)}
    for policy in ['S1', 'S2', 'S3']:
        candidates = scores.filter(pl.col('policy') == policy).sort(['net_pnl', 'variant_id'], descending=[True, False])
        if candidates.is_empty():
            raise ValueError(f'{policy}没有共同日期的完整费用后结果')
        selected.append(available[candidates['variant_id'][0]])
    scores.write_csv(output / 'development_scores.csv')
    write_json_atomic(output / 'selected_strategies.json', {'selected_on': 'val_dev_only', 'variants': selected})


# 【核心逻辑329】汇总冻结验证、压力测试及成交条件分析，所有失败保留为缺失任务。
def summarize_execution(filename: Path) -> None:
    """报告费用、现金、成交后价格变化和库存风险，比较只使用共同日期。"""
    config, source, output = research_context(filename)
    summary = output / 'summary'
    summary.mkdir(exist_ok=True)
    records = [json.loads(path.read_text()) for path in (output / 'cases').glob('*.json')]
    frame = pl.DataFrame(records, infer_schema_length=None)
    frame.write_parquet(summary / 'strategy_metrics.parquet', compression='zstd')
    frame.select(pl.exclude('signal_file')).write_csv(summary / 'strategy_metrics.csv')
    frozen = frame.filter(pl.col('date').is_in(config['dates']['val_select'])
                          & (pl.col('latency_ns') == 1_000_000) & (pl.col('quantity_btc') == .002))
    totals = frozen.group_by('policy', 'variant_id', 'queue_model').agg(
        pl.col('date').n_unique().alias('days'), pl.col('fills').sum(),
        pl.col('gross_trading_pnl_usdt').sum().alias('gross_pnl'),
        pl.col('trading_fees_usdt').sum().alias('fees'),
        pl.col('funding_payment_usdt').sum().alias('funding'),
        pl.col('net_pnl_after_funding_usdt').sum().alias('net_pnl'),
        pl.col('max_liquidation_drawdown_usdt').max().alias('worst_daily_drawdown'),
    ).sort('policy', 'queue_model')
    totals.write_csv(summary / 'frozen_validation.csv')
    probe_records = []
    for file in (output / 'probes').glob('*.parquet'):
        table = pl.read_parquet(file)
        if table.is_empty():
            continue
        date = table['date'][0]
        for signal in config['signals']:
            # 训练日只使用该日期之前训练的OOF模型；开发日使用冻结Train模型。
            if date in config['dates']['probe']:
                parts = [output / 'oof' / signal['id'] / fold / 'predictions.parquet' for fold in config['folds']]
                day_start = int(np.datetime64(date, 'ns').astype(np.int64))
                alpha = pl.scan_parquet(parts).filter(
                    pl.col('timestamp_ns').is_between(day_start - SECOND, day_start + DAY * SECOND, closed='left'),
                ).select('timestamp_ns', 'prediction_log_bps').collect(engine='streaming')
                calibration = json.loads((output / 'calibration.json').read_text())[signal['id']]
                alpha = alpha.with_columns((pl.col('prediction_log_bps') * calibration['slope'] + calibration['intercept']).alias('alpha'))
            else:
                alpha = pl.read_csv(output / 'signals' / f'{signal["id"]}_{date}.csv').rename({'prediction_bps': 'alpha'})
            joined = table.sort('decision_ts_ns').join_asof(
                alpha.select((pl.col('timestamp_ns') + config['prediction_latency_ns']).alias('ready_ns'), 'alpha').sort('ready_ns'),
                left_on='decision_ts_ns', right_on='ready_ns', strategy='backward', tolerance=SECOND,
            ).with_columns((pl.col('side') * pl.col('alpha')).alias('signed_alpha'))
            joined = joined.with_columns(pl.when(pl.col('signed_alpha') > 0).then(pl.lit('same_direction'))
                                        .when(pl.col('signed_alpha') <= 0).then(pl.lit('opposite_direction'))
                                        .otherwise(pl.lit('missing')).alias('condition'))
            labels = np.load(source / 'e0/y.npy', mmap_mode='r')
            start = json.loads((source / 'e0/manifest.json').read_text())['start_timestamp_ns']
            rows = ((joined['decision_ts_ns'].to_numpy() - start) // SECOND).astype('int64')
            index = int(np.flatnonzero(HORIZONS == 180)[0])
            joined = joined.with_columns(pl.Series('future_180s_signed_bps', labels[rows, index] * joined['side'].to_numpy()))
            probe_records.append(joined.with_columns(pl.lit(signal['id']).alias('signal')))
    if probe_records:
        opportunities = pl.concat(probe_records, how='diagonal_relaxed')
        opportunities.write_parquet(summary / 'conditional_opportunities.parquet', compression='zstd')
        opportunities.group_by('signal', 'condition', 'offset_ticks', 'ttl_ms').agg(
            pl.len().alias('opportunities'), pl.col('filled_any').mean().alias('fill_probability'),
            pl.col('future_180s_signed_bps').filter(pl.col('future_180s_signed_bps').is_finite()).mean().alias('unconditional_180s_bps'),
            pl.col('future_180s_signed_bps').filter(pl.col('filled_any') & pl.col('future_180s_signed_bps').is_finite()).mean().alias('filled_condition_180s_bps'),
            pl.col('markout_180s_bps').filter(pl.col('markout_180s_bps').is_finite()).mean(),
            pl.col('submitted_value_bps').mean(),
        ).write_csv(summary / 'conditional_fill_value.csv')
    status = json.loads((output / 'queue/status.json').read_text())
    failures = [name for name, task in status['tasks'].items() if task['status'] in ['failed', 'memory_stopped', 'skipped']]
    lines = ['# 预测信号与做市策略研究｜2026-11-10', '',
             '六组信号先通过训练期前向OOF校准，开发期选择参数，冻结验证期运行真实事件回放。',
             'Maker单侧2bps，Taker单侧5bps，资金费按官方记录结算。每日从空仓开始，结束前实际平仓。', '',
             '| 策略 | 队列 | 日期数 | 成交数 | 费用前损益 | 手续费 | 资金费 | 净收益USDT |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in totals.iter_rows(named=True):
        lines.append(f"| {row['policy']} | {row['queue_model']} | {row['days']} | {row['fills']} | {row['gross_pnl']:.4f} "
                     f"| {row['fees']:.4f} | {row['funding']:.4f} | {row['net_pnl']:.4f} |")
    lines.extend(['', '完整案例含库存风险、持有时间、强制退出成本及成交后100ms至300秒的价格变化。',
                  'strategy_metrics.csv包含延迟5ms/20ms及双倍规模压力场景。',
                  '冻结验证使用12个不连续历史日期，日末平仓，累计值为这些日期的独立账户损益合计。'])
    if failures:
        lines.extend(['', '未完成任务', *[f'- {name}' for name in failures]])
    (summary / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json_atomic(summary / 'completed.json', {'cases': len(records), 'failed_tasks': failures})
    if failures:
        raise RuntimeError('研究仍有失败任务，已在最终报告列出')


# 【非核心】将明确的执行阶段交给对应职责函数。
def run_execution_stage(filename: Path, stage: str, date: str | None) -> None:
    """日期任务独立重试，选参和汇总在相应依赖终态后执行。"""
    if stage == 'probe':
        run_probe_day(filename, date)
    elif stage in ['baseline', 'pilot', 'develop', 'validate', 'stress']:
        run_day_stage(filename, stage, date)
    elif stage == 'select':
        freeze_strategies(filename)
    else:
        summarize_execution(filename)
