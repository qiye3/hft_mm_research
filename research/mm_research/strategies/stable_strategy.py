"""以跨周和跨月稳定性选择策略，完整保留极端行情与失败候选。"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import context, SECOND
from mm_research.multihorizon.inventory import funding_inputs, simulate_inventory
from mm_research.strategies.profit_research import save_screening_case


# 【核心逻辑384】先等权组合方向价值，再选择净价值为正的一侧，不按复核表现挑种子。
def combined_signal(config: dict, group: str, split: str) -> np.ndarray:
    """每个来源提供买卖两头；agreement要求全部来源对同一方向有正价值。"""
    specification = config['groups'][group]
    edges = []
    for source in specification['sources']:
        folder = Path(source['folder']) / 'models'
        buy = np.load(folder / (source['pattern'].format(side=1)) / f'{split}_net_edge_bps.npy')
        sell = np.load(folder / (source['pattern'].format(side=-1)) / f'{split}_net_edge_bps.npy')
        edges.append(np.column_stack([buy, sell]))
    combined = combine_values(np.array(edges), specification['combination'])
    buy, sell = combined[:, 0], combined[:, 1]
    return np.where(np.maximum(buy, sell) > 0., np.where(buy >= sell, 1., -1.), 0.)


# 【非核心】固定组合规则抑制单模型极值，分歧惩罚是启发式评分而非统计置信区间。
def combine_values(edges: np.ndarray, method: str) -> np.ndarray:
    """输入模型×时间×方向价值，输出同一时间与方向上的固定组合评分。"""
    average = np.mean(edges, axis=0)
    if method == 'mean':
        return average
    if method == 'long_only':
        # 方向消融在发单前禁用空头，保留多头原始价值，不事后删除亏损交易。
        average[:, 1] = -np.inf
        return average
    if method == 'median':
        return np.median(edges, axis=0)
    if method == 'dispersion_penalty':
        return average - np.std(edges, axis=0)
    if method == 'agreement':
        agrees = np.all(edges > 0., axis=0)
        return np.where(agrees, average, -np.inf)
    raise ValueError(f'未知价值组合规则：{method}')


# 【非核心】用Polars滑窗计算严格落在决策前的600秒波动，不复制事件窗口矩阵。
def visible_management(market: np.ndarray, begin: int, stop: int) -> np.ndarray:
    """输出本地买卖价和完整600收益窗口；跨质量边界或缺价保留NaN。"""
    prefix = begin - 601
    values = np.asarray(market[prefix:stop])
    frame = pl.DataFrame({'mid': values[:, 0], 'valid': values[:, 8], 'segment': values[:, 9]})
    valid = pl.col('mid').is_finite() & (pl.col('mid') > 0) & (pl.col('valid') > .5)
    frame = frame.with_columns(pl.when(valid).then(pl.col('mid').log()).alias('log_price'))
    frame = frame.with_columns(pl.col('log_price').diff().pow(2).alias('square_return'))
    frame = frame.with_columns(
        (pl.col('square_return').rolling_sum(600, min_samples=600).shift(1).sqrt() * 10000).alias('vol'),
        pl.col('segment').rolling_min(601).shift(1).alias('minimum_segment'),
        pl.col('segment').rolling_max(601).shift(1).alias('maximum_segment'),
    )
    volatility = frame.select(pl.when(pl.col('minimum_segment') == pl.col('maximum_segment'))
                              .then(pl.col('vol')).alias('vol'))['vol'].to_numpy()[601:]
    return np.column_stack([market[begin:stop, 1], market[begin:stop, 2], volatility])


# 【非核心】完整日级权益覆盖无成交日期，再按UTC日历周和月汇总。
def stability_metrics(output: Path, identifier: str, start_seconds: float) -> dict:
    """从连续账户日末权益计算最差周/月和剔除最好周收益，避免只看平仓日。"""
    daily = pl.read_csv(output / 'cases' / f'{identifier}.daily.csv')
    dates = [datetime.fromtimestamp(start_seconds + index * 86400, timezone.utc).date()
             for index in daily['day_index'].to_list()]
    daily = daily.with_columns(pl.Series('date', dates))
    weekly = daily.group_by(pl.col('date').dt.truncate('1w').alias('period')).agg(pl.col('net_pnl_usdt').sum()).sort('period')
    monthly = daily.group_by(pl.col('date').dt.strftime('%Y-%m').alias('period')).agg(pl.col('net_pnl_usdt').sum()).sort('period')
    daily.write_csv(output / 'cases' / f'{identifier}.daily.csv')
    weekly.write_csv(output / 'cases' / f'{identifier}.weekly.csv')
    monthly.write_csv(output / 'cases' / f'{identifier}.monthly.csv')
    return {
        'worst_week_usdt': float(weekly['net_pnl_usdt'].min()),
        'worst_month_usdt': float(monthly['net_pnl_usdt'].min()),
        'positive_weeks': int((weekly['net_pnl_usdt'] > 0).sum()),
        'total_weeks': weekly.height,
        'excluding_best_week_usdt': float(daily['net_pnl_usdt'].sum() - weekly['net_pnl_usdt'].max()),
    }


# 【核心逻辑385】同一方向信号比较固定持仓、信号失效和事前波动保护，账本复用已有执行器。
def screen_stability(filename: Path, group: str, split: str) -> None:
    """开发运行预先定义规则，历史复核仅运行已冻结选择和固定基准。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    bounds = json.loads((source / 'e0/splits.json').read_text())['segments'][split]
    begin, stop = bounds['start_row'], bounds['stop_row']
    signal = combined_signal(config, group, split)
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    management = visible_management(market, begin, stop)
    books = np.load(source / f'e3/{split}_books.npy', mmap_mode='r')
    start_seconds = manifest['start_timestamp_ns'] / SECOND + begin
    times = start_seconds + np.arange(stop - begin) + .002
    funding = funding_inputs(source, manifest, begin, stop)
    if split == 'val_dev':
        rules = list(config['rules'])
    else:
        frozen = json.loads((output / 'selected.json').read_text())
        rules = [item['rule'] for item in frozen['candidates'] if item['group'] == group]
    for rule in rules:
        identifier = f'{group}_{rule}_{split}'
        destination = output / 'cases' / f'{identifier}.json'
        if destination.exists() and 'worst_week_usdt' in json.loads(destination.read_text()):
            continue
        specification = config['rules'][rule]
        direction = signal.copy()
        entry_allowed = np.ones(len(signal))
        entry_allowed[:120] = 0.
        entry_allowed[-specification['max_hold_s'] - 20:] = 0.
        if specification['policy'] == 0:
            direction[entry_allowed == 0.] = 0.
        managed_inputs = np.column_stack([management, entry_allowed])
        ledger, fills, position = simulate_inventory(
            direction[:, None], books, times, np.ones(1), specification['policy'], 0,
            specification['minimum_hold_s'], 1., 0., 1., .002, .0005, 100., .001,
            specification['max_hold_s'], 10000., *funding, managed_inputs,
        )
        record = save_screening_case(output, identifier, {
            'group': group, 'rule': rule, 'split': split, 'taker_fee_bps': 5,
            'execution_type': 'arrival_snapshot_depth_walk', **specification,
        }, ledger, fills, position, funding)
        record.update(stability_metrics(output, identifier, start_seconds))
        write_json_atomic(destination, record)


# 【核心逻辑386】开发期按最差月和剔除最好周排序冻结三例，复核结果不参与选择。
def select_stable(filename: Path) -> None:
    """保留固定基准；门槛要求正收益、剔除最好周仍正、至少20笔和10个活动日。"""
    _, config, output = context(filename)
    rows = [json.loads(file.read_text()) for file in (output / 'cases').glob('*_val_dev.json')]
    expected = len(config['groups']) * len(config['rules'])
    if len(rows) != expected:
        raise ValueError(f'开发实验尚未完整：{len(rows)}/{expected}')
    table = pl.DataFrame(rows).sort(
        ['worst_month_usdt', 'excluding_best_week_usdt', 'max_drawdown_usdt'],
        descending=[True, True, False],
    )
    table.write_csv(output / 'development.csv')
    eligible = table.filter(
        (pl.col('net_pnl_usdt') > 0) & (pl.col('excluding_best_week_usdt') > 0)
        & (pl.col('round_trips') >= 20) & (pl.col('active_days') >= 10)
    ).head(3).select('group', 'rule').to_dicts()
    qualified_count = len(eligible)
    baseline = {'group': 'cost1800', 'rule': 'fixed7200'}
    if baseline not in eligible:
        eligible.append(baseline)
    write_json_atomic(output / 'selected.json', {
        'selection': 'val_dev_only_worst_month_then_excluding_best_week_then_drawdown',
        'candidates': eligible, 'qualified_count': qualified_count,
    })


# 【核心逻辑387】冻结后执行历史复核并形成完整对照，不按复核重新排列选择。
def review_stable(filename: Path) -> None:
    """汇总所有开发失败案例与实际复核结果，未通过稳定性门槛如实保留。"""
    _, config, output = context(filename)
    selected = json.loads((output / 'selected.json').read_text())
    for group in sorted({case['group'] for case in selected['candidates']}):
        screen_stability(filename, group, 'val_select')
    table = pl.DataFrame([json.loads(file.read_text()) for file in (output / 'cases').glob('*.json')])
    table.sort(['split', 'group', 'rule']).write_csv(output / 'strategies.csv')
    lines = ['# 跨市场环境稳定性实验', '',
             '开发：2026-03-23—05-14；历史复核：05-15—07-06。每笔0.002BTC，Taker开平各5bps，另计资金费。',
             '先完成全部开发规则再冻结选择，复核只运行入选规则与原候选。结果属于快照执行估计。', '',
             '| 分区 | 模型 | 退出规则 | 净收益USDT | 最差月 | 最差周 | 去最好周 | 回撤 | 往返 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in table.sort(['split', 'group', 'rule']).iter_rows(named=True):
        lines.append(f"|{row['split']}|{row['group']}|{row['rule']}|{row['net_pnl_usdt']:.4f}|{row['worst_month_usdt']:.4f}|{row['worst_week_usdt']:.4f}|{row['excluding_best_week_usdt']:.4f}|{row['max_drawdown_usdt']:.4f}|{row['round_trips']}|")
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json_atomic(output / 'completed.json', {'status': 'success', 'selected': selected})
