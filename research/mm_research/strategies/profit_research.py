"""跨市场组合信号的费用后交易研究，选择只读取开发区间，复核规则冻结。"""

import fcntl
import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import context, HORIZONS, SECOND, DAY
from mm_research.multihorizon.inventory import funding_inputs, simulate_inventory
from mm_research.strategies.tradeability_execution import snapshot_round_trips


# 【核心逻辑354】仅用原训练尾部的预测分布定义强信号门槛，三个种子等权平均。
def freeze_thresholds(filename: Path) -> None:
    """分位点只读取Train内部tune输入，任何开发或复核标签都不参与门槛计算。"""
    import cupy as cp
    import xgboost as xgb

    _, config, output = context(filename)
    destination = output / 'thresholds.json'
    if destination.exists():
        return
    model_root = Path(config['models'])
    source = Path(config['source'])
    preprocessing = np.load(source / 'e1/folds/final/preprocessing.npz')
    index = int(np.flatnonzero(HORIZONS == 180)[0])
    target_mean = float(preprocessing['target_mean'][index])
    target_scale = float(preprocessing['target_scale'][index])
    old = np.load(Path(config['old_inputs']) / 'tune_x.npy', mmap_mode='r')
    results = {}
    with Path('/tmp/hft_mm_research_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for group in config['groups']:
            added = np.load(model_root / group / 'inputs/180s/tune_x.npy', mmap_mode='r')
            prediction = np.zeros(len(old), dtype='float64')
            for seed in [17, 42, 73]:
                model = xgb.Booster(model_file=str(model_root / group / f'models/180s_combined_xgb_seed{seed}/model.ubj'))
                model.set_param({'device': 'cuda', 'nthread': 2})
                for begin in range(0, len(old), 65536):
                    stop = min(begin + 65536, len(old))
                    values = cp.asarray(np.concatenate([old[begin:stop], added[begin:stop]], axis=1))
                    prediction[begin:stop] += cp.asnumpy(model.inplace_predict(values)) / 3
            prediction = prediction * target_scale + target_mean
            results[group] = {
                str(quantile): float(np.quantile(np.abs(prediction), quantile))
                for quantile in config['quantiles']
            }
            print('TRAIN_THRESHOLDS', group, results[group], flush=True)
    write_json_atomic(destination, {'fit': 'train_internal_tune_prediction_distribution', 'groups': results})


# 【非核心】从真实成交与资金费形成完整往返和日级净收益，末端必须实际平仓。
def save_screening_case(output: Path, identifier: str, specification: dict,
                        ledger: np.ndarray, fills: np.ndarray, position: float, funding: tuple) -> dict:
    """输出与实际交易现金流核对后的指标，连续账户按UTC日切分损益变化。"""
    if abs(position) >= .0005:
        raise ValueError(f'区间结束仍有仓位 {identifier}')
    trades = snapshot_round_trips(fills, funding)
    valid_equity = ledger[:, 7][np.isfinite(ledger[:, 7])]
    net = float(valid_equity[-1])
    if abs(float(trades['net_pnl_usdt'].sum()) - net) > 1e-6:
        raise ValueError(f'逐笔与账户损益不一致 {identifier}')
    daily = []
    previous = 0.
    for begin in range(0, len(ledger), DAY):
        current = ledger[begin:begin + DAY, 7]
        current = current[np.isfinite(current)]
        value = float(current[-1])
        daily.append({'day_index': begin // DAY, 'net_pnl_usdt': value - previous})
        previous = value
    folder = output / 'cases'
    folder.mkdir(exist_ok=True)
    trades.write_parquet(folder / f'{identifier}.trades.parquet', compression='zstd')
    # 分钟抽样直接来自执行账本，不根据往返成交重建或插值权益。
    sampled = ledger[::60]
    pl.DataFrame({
        'timestamp_s': sampled[:, 0],
        'position_btc': sampled[:, 3],
        'cashflow_usdt': sampled[:, 4],
        'fees_usdt': sampled[:, 5],
        'funding_paid_usdt': sampled[:, 6],
        'net_pnl_liquidated_usdt': sampled[:, 7],
        'margin_rejections': sampled[:, 10],
    }).write_parquet(folder / f'{identifier}.timeline.parquet', compression='zstd')
    pl.DataFrame(daily).write_csv(folder / f'{identifier}.daily.csv')
    record = {
        'id': identifier, **specification, 'round_trips': trades.height,
        'gross_pnl_usdt': float(trades['gross_pnl_usdt'].sum()),
        'fees_usdt': float(trades['fees_usdt'].sum()),
        'funding_paid_usdt': float(trades['funding_paid_usdt'].sum()),
        'net_pnl_usdt': net,
        'max_drawdown_usdt': float((np.maximum.accumulate(np.r_[0., valid_equity])[1:] - valid_equity).max()),
        'active_days': int(np.count_nonzero(pl.DataFrame(daily)['net_pnl_usdt'].to_numpy())),
        'excluding_best_day_usdt': net - max(item['net_pnl_usdt'] for item in daily),
        'final_position_btc': position,
    }
    write_json_atomic(folder / f'{identifier}.json', record)
    print('PNL', identifier, net, trades.height, flush=True)
    return record


# 【核心逻辑355】用同一180秒模型比较强信号门槛和退出时长，交易按真实到达盘口逐档执行。
def screen_group(filename: Path, group: str, split: str, signal_column: str = 'prediction_log_bps') -> None:
    """开发只比较预先定义的九种规则，复核只运行按开发净收益冻结的规则。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    bounds = json.loads((source / 'e0/splits.json').read_text())['segments'][split]
    signals = pl.read_parquet(Path(config['models']) / 'signals' / f'{group}_{split}.parquet')
    books = np.load(source / f'e3/{split}_books.npy', mmap_mode='r')
    funding = funding_inputs(source, manifest, bounds['start_row'], bounds['stop_row'])
    times = signals['timestamp_ns'].to_numpy().astype('float64') / SECOND + .002
    alpha = signals[signal_column].to_numpy()
    thresholds = json.loads((output / 'thresholds.json').read_text())['groups'][group]
    if split == 'val_dev':
        cases = [{'quantile': quantile, 'hold_s': hold} for quantile in config['quantiles'] for hold in config['holds_s']]
    else:
        cases = [case for case in json.loads((output / 'selected.json').read_text())['candidates'] if case['group'] == group]
    seen_rules = set()
    for case in cases:
        quantile = case['quantile']
        hold = case['hold_s']
        identity = (thresholds[str(quantile)], hold)
        if identity in seen_rules:
            continue
        seen_rules.add(identity)
        identifier = f'{group}_q{quantile}_h{hold}_{split}'
        if (output / 'cases' / f'{identifier}.json').exists():
            continue
        eligible = np.isfinite(alpha) & (np.abs(alpha) >= thresholds[str(quantile)])
        eligible[:120] = False
        eligible[-hold - 20:] = False
        # P0只访问单个方向列，复用已有执行器即可，无需分配完整16周期矩阵。
        direction = np.where(eligible, np.sign(alpha), 0.).reshape(-1, 1)
        ledger, fills, position = simulate_inventory(
            direction, books, times, np.ones(1), 0, 0, hold,
            1., 0., 1., .002, .0005, 100., .001, hold, 10000., *funding,
        )
        save_screening_case(output, identifier, {
            'group': group, 'split': split, 'quantile': quantile, 'hold_s': hold,
            'threshold_bps': thresholds[str(quantile)], 'taker_fee_bps': 5,
            'execution_type': 'arrival_time_snapshot_depth_walk', 'account_reset': 'segment_start_only',
        }, ledger, fills, position, funding)


# 【核心逻辑356】每个信息组只保留开发净收益最高的正收益规则，之后冻结其复核参数。
def select_rules(filename: Path) -> None:
    """完整开发候选保存下来，后续不能按复核收益更换已选规则。"""
    _, config, output = context(filename)
    rows = [json.loads(file.read_text()) for file in (output / 'cases').glob('*_val_dev.json')]
    table = pl.DataFrame(rows).sort('net_pnl_usdt', descending=True)
    table.write_csv(output / 'development.csv')
    candidates = table.filter((pl.col('net_pnl_usdt') > 0) & (pl.col('round_trips') > 0)).unique(
        subset='group', keep='first', maintain_order=True,
    ).select('group', 'quantile', 'hold_s').to_dicts()
    write_json_atomic(output / 'selected.json', {'selection': 'val_dev_only', 'candidates': candidates})


# 【核心逻辑357】汇总费用后开发及冻结复核收益，正收益候选随后进入原生订单回放。
def report_profit(filename: Path) -> None:
    """报告全部开发和实际复核记录，未通过时保留可定位的负收益原因。"""
    _, config, output = context(filename)
    rows = [json.loads(file.read_text()) for file in (output / 'cases').glob('*.json')]
    table = pl.DataFrame(rows).sort(['split', 'net_pnl_usdt'], descending=[False, True])
    summary = output / 'summary'
    summary.mkdir(exist_ok=True)
    table.write_csv(summary / 'strategies.csv')
    passed = table.filter((pl.col('split') == 'val_select') & (pl.col('net_pnl_usdt') > 0))
    lines = [
        '# 跨市场信号费用后交易实验｜2026-11-11', '',
        '信号预测目标由上游模型配置声明，交易退出时长比较' + '/'.join(str(value) for value in config['holds_s']) + '秒。门槛由原Train内部预测分布冻结。',
        '开发区间固定为2026-03-23—05-14，历史复核为05-15—07-06。复核按selected.json中预先冻结的规则执行；该文件记录开发选参或固定规则种子对照的来源。',
        '账户跨日连续，每笔0.002BTC，主动开平各5bps手续费，逐档盘口执行并计资金费，区间末尾真实退出。', '',
        '| 信息组 | 分区 | 分位 | 持有秒数 | 往返数 | 费用前USDT | 费用USDT | 净收益USDT |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in table.iter_rows(named=True):
        lines.append(f"| {row['group']} | {row['split']} | {row['quantile']} | {row['hold_s']} | {row['round_trips']} | {row['gross_pnl_usdt']:.3f} | {row['fees_usdt']:.3f} | {row['net_pnl_usdt']:.3f} |")
    lines.extend(['', f'历史复核净收益为正的候选数量为{passed.height}，后续执行验证使用原生订单框架。',
                  '完整逐笔往返和日级权益变化保存在cases，策略参数及最大回撤见strategies.csv。'])
    (summary / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json_atomic(output / 'completed.json', {'status': 'success', 'positive_snapshot_candidates': passed.to_dicts()})
