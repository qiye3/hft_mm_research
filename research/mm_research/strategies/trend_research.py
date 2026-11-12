"""低换手趋势与反转对照，所有决策只读取过去收益，手续费及资金费逐笔核算。"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import context, SECOND
from mm_research.multihorizon.inventory import funding_inputs, simulate_inventory
from mm_research.strategies.profit_research import save_screening_case


# 【核心逻辑369】比较过去收益的趋势延续与反转，保持同一风险单位和真实到达盘口执行。
def screen_trend(filename: Path, family: str, split: str) -> None:
    """每类开发12套粗网格，冻结后只复核该类开发净收益最高的正收益规则。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    bounds = json.loads((source / 'e0/splits.json').read_text())['segments'][split]
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    rows = np.arange(bounds['start_row'], bounds['stop_row'])
    books = np.load(source / f'e3/{split}_books.npy', mmap_mode='r')
    times = (manifest['start_timestamp_ns'] + rows * SECOND).astype('float64') / SECOND + .002
    funding = funding_inputs(source, manifest, bounds['start_row'], bounds['stop_row'])
    if split == 'val_dev':
        cases = [
            {'lookback_s': lookback, 'entry_bps': entry, 'hold_s': hold}
            for lookback in config['lookbacks_s'] for entry in config['entry_bps'] for hold in config['holds_s']
        ]
    else:
        cases = [case for case in json.loads((output / 'selected.json').read_text())['candidates'] if case['family'] == family]
    for case in cases:
        lookback = case['lookback_s']
        entry = case['entry_bps']
        hold = case['hold_s']
        identifier = f'{family}_l{lookback}_e{entry}_h{hold}_{split}'
        if (output / 'cases' / f'{identifier}.json').exists():
            continue
        current = np.asarray(market[rows - 1, 0])
        previous = np.asarray(market[rows - 1 - lookback, 0])
        returns = np.full(len(rows), np.nan)
        valid = np.isfinite(current) & np.isfinite(previous) & (current > 0) & (previous > 0)
        returns[valid] = np.log(current[valid] / previous[valid]) * 10000
        eligible = np.isfinite(returns) & (np.abs(returns) >= entry)
        eligible[:120] = False
        eligible[-hold - 20:] = False
        direction = np.where(eligible, np.sign(returns), 0.)
        if family == 'reversal':
            direction = -direction
        ledger, fills, position = simulate_inventory(
            direction.reshape(-1, 1), books, times, np.ones(1), 0, 0, hold,
            1., 0., 1., .002, .0005, 100., .001, hold, 10000., *funding,
        )
        save_screening_case(output, identifier, {
            'family': family, 'split': split, 'lookback_s': lookback,
            'entry_bps': entry, 'hold_s': hold, 'taker_fee_bps': 5,
            'execution_type': 'arrival_time_snapshot_depth_walk', 'account_reset': 'segment_start_only',
        }, ledger, fills, position, funding)


# 【核心逻辑370】每类只按开发区间的实际净收益冻结一套规则。
def select_trend(filename: Path) -> None:
    """两个交易假设分别展示，复核成绩不反向参与参数选择。"""
    _, config, output = context(filename)
    records = [json.loads(file.read_text()) for file in (output / 'cases').glob('*_val_dev.json')]
    table = pl.DataFrame(records).sort('net_pnl_usdt', descending=True)
    table.write_csv(output / 'development.csv')
    candidates = table.filter((pl.col('net_pnl_usdt') > 0) & (pl.col('round_trips') > 0)).unique(
        subset='family', keep='first', maintain_order=True,
    ).select('family', 'lookback_s', 'entry_bps', 'hold_s').to_dicts()
    write_json_atomic(output / 'selected.json', {'selected_on': 'val_dev_only', 'candidates': candidates})


# 【核心逻辑371】汇总低换手策略的完整账户损益及冻结复核结果。
def report_trend(filename: Path) -> None:
    """输出全部规则、费用、资金费与回撤，继续保留与原生订单验证的执行区别。"""
    _, config, output = context(filename)
    records = [json.loads(file.read_text()) for file in (output / 'cases').glob('*.json')]
    table = pl.DataFrame(records).sort('split', 'net_pnl_usdt', descending=[False, True])
    summary = output / 'summary'
    summary.mkdir(exist_ok=True)
    table.write_csv(summary / 'strategies.csv')
    lines = [
        '# 低换手趋势与反转费用后对照｜2026-11-11', '',
        '仅使用上一完整秒已经发生的过去收益。比较1小时/4小时/1日观察窗、50/150bps入场幅度、6小时/1日持有。',
        '每笔0.002BTC，主动开平各5bps，计资金费及真实十档快照执行，账户跨日连续。开发选参数后冻结到历史复核。', '',
        '| 类型 | 分区 | 观察秒数 | 入场bps | 持有秒数 | 往返数 | 净收益USDT | 剔除最佳日USDT |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in table.iter_rows(named=True):
        lines.append(f"| {row['family']} | {row['split']} | {row['lookback_s']} | {row['entry_bps']} | {row['hold_s']} | {row['round_trips']} | {row['net_pnl_usdt']:.3f} | {row['excluding_best_day_usdt']:.3f} |")
    (summary / 'report.md').write_text('\n'.join(lines) + '\n')
    passed = table.filter((pl.col('split') == 'val_select') & (pl.col('net_pnl_usdt') > 0))
    write_json_atomic(output / 'completed.json', {'status': 'success', 'positive_snapshot_candidates': passed.to_dicts()})
