"""固定账户本金下重新逐档回放仓位规模，报告费用后年化与实际敞口。"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import context, SECOND
from mm_research.multihorizon.inventory import funding_inputs, simulate_inventory
from mm_research.strategies.profit_research import save_screening_case
from mm_research.strategies.stable_strategy import combined_signal, visible_management, stability_metrics


# 【非核心】账户收益率使用全部日历时间与固定本金，不以单笔保证金缩小分母。
def account_metrics(ledger: np.ndarray, books: np.ndarray, capital: float) -> dict:
    """逐时权益已扣费用和资金费，敞口按当时盘口中价除以实际净权益。"""
    equity = capital + ledger[:, 7]
    valid = np.isfinite(equity)
    final_equity = float(equity[valid][-1])
    duration_days = (ledger[-1, 0] - ledger[0, 0] + 1.) / 86400.
    total_return = final_equity / capital - 1.
    annualized = (final_equity / capital) ** (365. / duration_days) - 1.
    marked_exposure = np.abs(ledger[:, 3]) * (books[:, 0] + books[:, 20]) * .5
    leverage = marked_exposure[valid] / equity[valid]
    peak_equity = np.maximum.accumulate(np.r_[capital, equity[valid]])[1:]
    drawdown = 1. - equity[valid] / peak_equity
    return {
        'capital_usdt': capital, 'duration_days': duration_days,
        'total_return': total_return, 'annualized_return': annualized,
        'max_drawdown_fraction': float(drawdown.max()),
        'peak_gross_leverage': float(np.nanmax(leverage)),
        'entry_margin_rejections': int(ledger[-1, 10]),
        'annualization': 'calendar_365_compound_fixed_initial_capital',
    }


# 【核心逻辑413】只用截至当前秒连续同向的信号决定入场资格，不改变持仓中的退出规则。
def confirmed_entry(signal: np.ndarray, seconds: int) -> np.ndarray:
    """完整秒网格上零信号或方向变化打断确认；前缀不足时不准入场。"""
    if seconds < 1:
        raise ValueError('信号确认秒数必须为正整数')
    rows = np.arange(len(signal))
    changes = np.r_[True, signal[1:] != signal[:-1]]
    run_start = np.maximum.accumulate(np.where(changes, rows, 0))
    return (signal != 0.) & (rows - run_start + 1 >= seconds)


# 【核心逻辑416】短周期预测只作为入场方向确认，严格校对整秒时间键，不使用未来标签。
def directional_entry_filter(filename: Path, signal: np.ndarray, start_ns: int) -> np.ndarray:
    """读取当时已生成的预测，正负方向与主信号相同才准入；缺失或零预测不准入。"""
    frame = pl.read_parquet(filename, columns=['timestamp_ns', 'prediction_log_bps'])
    expected = start_ns + np.arange(len(signal), dtype='int64') * SECOND
    if not np.array_equal(frame['timestamp_ns'].to_numpy(), expected):
        raise ValueError('短周期确认预测与主策略完整时间网格不一致')
    prediction = frame['prediction_log_bps'].to_numpy()
    return np.isfinite(prediction) & (prediction * signal > 0.)


# 【核心逻辑391】每个规模重新按十档数量撮合，现金约束与手续费随实际成交变化。
def screen_account(filename: Path, group: str, split: str) -> None:
    """开发比较固定规模网格，复核只使用开发冻结规模，不线性乘大旧利润。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    bounds = json.loads((source / 'e0/splits.json').read_text())['segments'][split]
    begin, stop = bounds['start_row'], bounds['stop_row']
    predictions = combined_signal(config, group, split)
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    management = visible_management(market, begin, stop)
    books = np.load(source / f'e3/{split}_books.npy', mmap_mode='r')
    start_seconds = manifest['start_timestamp_ns'] / SECOND + begin
    times = start_seconds + np.arange(stop - begin) + .002
    funding = funding_inputs(source, manifest, begin, stop)
    rule = config['rules'][group]
    if split == 'val_dev':
        quantities = config['quantities_btc']
    else:
        chosen = json.loads((output / 'selected.json').read_text())['candidates']
        quantities = [row['quantity_btc'] for row in chosen if row['group'] == group]
    entry_allowed = np.ones(len(predictions))
    if 'entry_filter_prediction' in rule:
        filter_file = Path(rule['entry_filter_prediction'].format(split=split))
        entry_allowed *= directional_entry_filter(
            filter_file, predictions, manifest['start_timestamp_ns'] + begin * SECOND,
        )
    if 'entry_confirmation_s' in rule:
        entry_allowed *= confirmed_entry(predictions, rule['entry_confirmation_s'])
    entry_allowed[:120] = 0.
    entry_allowed[-rule['max_hold_s'] - 20:] = 0.
    if rule['policy'] == 0:
        predictions[entry_allowed == 0.] = 0.
    managed_inputs = np.column_stack([management, entry_allowed])
    for quantity in quantities:
        identifier = f'{group}_q{quantity}_{split}'
        destination = output / 'cases' / f'{identifier}.json'
        if destination.exists() and 'annualized_return' in json.loads(destination.read_text()):
            continue
        ledger, fills, position = simulate_inventory(
            predictions[:, None], books, times, np.ones(1), rule['policy'], 0,
            rule['minimum_hold_s'], 1., 0., 1., quantity, .0005, 100., .001,
            rule['max_hold_s'], config['capital_usdt'], *funding, managed_inputs,
        )
        result = save_screening_case(output, identifier, {
            'group': group, 'split': split, 'quantity_btc': quantity, **rule,
            'execution_type': 'arrival_snapshot_depth_walk', 'taker_fee_bps': 5,
        }, ledger, fills, position, funding)
        result.update(stability_metrics(output, identifier, start_seconds))
        result.update(account_metrics(ledger, books, config['capital_usdt']))
        result['within_exposure_limit'] = result['peak_gross_leverage'] <= config['max_gross_leverage']
        write_json_atomic(destination, result)


# 【核心逻辑392】只按开发选择达到10%年化且敞口合规的最小规模，冻结后执行历史复核。
def select_and_review_account(filename: Path) -> None:
    """开发未达标的模型不进入复核；完整日历和收益集中度指标始终保留。"""
    _, config, output = context(filename)
    records = [json.loads(file.read_text()) for file in (output / 'cases').glob('*_val_dev.json')]
    if len(records) != len(config['groups']) * len(config['quantities_btc']):
        raise ValueError('仓位规模开发实验尚未完整')
    table = pl.DataFrame(records)
    candidates = table.filter(
        pl.col('within_exposure_limit') & (pl.col('annualized_return') > config['target_annualized_return'])
        & (pl.col('excluding_best_week_usdt') > 0)
    ).sort('quantity_btc').unique(subset='group', keep='first', maintain_order=True)
    write_json_atomic(output / 'selected.json', {
        'selection': 'val_dev_only_smallest_quantity_above_target_and_positive_without_best_week',
        'candidates': candidates.select('group', 'quantity_btc').to_dicts(),
    })
    for group in candidates['group'].to_list():
        screen_account(filename, group, 'val_select')
    table = pl.DataFrame([json.loads(file.read_text()) for file in (output / 'cases').glob('*.json')])
    table.sort(['split', 'group', 'quantity_btc']).write_csv(output / 'strategies.csv')
    write_json_atomic(output / 'completed.json', {'status': 'success'})
