"""宽价差报价的原生订单实验，开发期选一套规则后冻结到既定复核日期。"""

import json
from pathlib import Path

import polars as pl

from mm_research.data.storage import file_locks, write_json_atomic
from mm_research.strategies.forecast_execution import prepare_execution_day, run_variant, variant_name
from mm_research.strategies.forecast_research import research_context


# 【非核心】固定比较覆盖费用的三个报价距离与两档库存持有上限。
def wide_variants(config: dict) -> list[dict]:
    """返回六套规则，日期与成交结果不参与生成候选。"""
    return [
        {
            'signal_id': 'zero', 'mode': 'fixed', 'policy': 'wide',
            'width_bps': width, 'ttl_ms': 1000, 'hold_s': hold, 'dual': False,
        }
        for width in config['widths_bps'] for hold in config['holds_s']
    ]


# 【核心逻辑358】宽报价直接交给既有原生挂撤单、排队、成交和库存模块执行。
def run_wide_day(filename: Path, date: str, split: str) -> None:
    """每个交易日独立空仓开始并实际平仓，日收益以完整账本计手续费和资金费。"""
    config, source, output = research_context(filename)
    if split == 'val_dev':
        variants = wide_variants(config)
    else:
        variants = json.loads((output / 'selected.json').read_text())['candidates']
    execute_quote_day(config, source, output, date, split, variants)


# 【非核心】各报价实验共用逐日准备、执行、账本及独立恢复位置。
def execute_quote_day(config: dict, source: Path, output: Path, date: str,
                      split: str, variants: list[dict]) -> None:
    """复用已有执行器，保存每个真实回放的费用后损益。"""
    for name, origin in [('l2', source / 'l2'), ('build', Path(config['binary_folder']))]:
        destination = output / name
        with file_locks([destination]):
            if not destination.exists():
                destination.symlink_to(origin.resolve(), target_is_directory=True)
    prepare_execution_day(config, source, output, date)
    records = []
    for variant in variants:
        result = run_variant(config, source, output, date, variant, 'risk_adverse')
        result['split'] = split
        records.append(result)
        print('NATIVE_PNL', date, variant_name(variant), result['net_pnl_after_funding_usdt'], flush=True)
    folder = output / 'daily'
    folder.mkdir(exist_ok=True)
    write_json_atomic(folder / f'{split}_{date}.json', {'records': records})


# 【核心逻辑359】按12个开发日累计费用后净收益选择一套正收益宽报价规则。
def select_wide(filename: Path) -> None:
    """仅用开发日期，所有失败由队列明确阻止选择阶段。"""
    config, source, output = research_context(filename)
    records = []
    for date in config['dates']['val_dev']:
        records.extend(json.loads((output / 'daily' / f'val_dev_{date}.json').read_text())['records'])
    totals = pl.DataFrame(records).group_by('variant_id').agg(
        pl.col('net_pnl_after_funding_usdt').sum().alias('net_pnl_usdt'),
        pl.col('fills').sum(), pl.col('date').n_unique().alias('days'),
    ).sort('net_pnl_usdt', descending=True)
    totals.write_csv(output / 'development.csv')
    selected = totals.filter((pl.col('net_pnl_usdt') > 0) & (pl.col('fills') > 0)).head(1)
    candidates = []
    if selected.height:
        name = selected['variant_id'][0]
        fields = ['signal_id', 'mode', 'policy', 'width_bps', 'ttl_ms', 'hold_s', 'dual']
        chosen = next(record for record in records if record['variant_id'] == name)
        candidates = [{field: chosen[field] for field in fields}]
    write_json_atomic(output / 'selected.json', {'selected_on': 'val_dev_only', 'candidates': candidates})


# 【核心逻辑360】汇总原生真实成交的费用后结果，保留独立日账户口径及完整账本路径。
def report_wide(filename: Path, description: str = '开发12日比较报价半宽8/16/32bps、库存持有上限180/1800秒，按开发累计净收益冻结一套正收益规则后运行12个历史复核日。') -> None:
    """成功结果必须包含两段均为正的同一冻结规则，失败实验继续保留。"""
    config, source, output = research_context(filename)
    records = []
    for file in (output / 'daily').glob('*.json'):
        records.extend(json.loads(file.read_text())['records'])
    table = pl.DataFrame(records)
    totals = table.group_by('split', 'variant_id').agg(
        pl.col('net_pnl_after_funding_usdt').sum().alias('net_pnl_usdt'),
        pl.col('gross_trading_pnl_usdt').sum().alias('gross_pnl_usdt'),
        pl.col('trading_fees_usdt').sum().alias('fees_usdt'),
        pl.col('funding_payment_usdt').sum().alias('funding_paid_usdt'),
        pl.col('fills').sum(), pl.col('date').n_unique().alias('days'),
        pl.col('net_pnl_after_funding_usdt').min().alias('worst_day_usdt'),
        (pl.col('net_pnl_after_funding_usdt').sum() - pl.col('net_pnl_after_funding_usdt').max()).alias('excluding_best_day_usdt'),
    ).sort('split', 'net_pnl_usdt', descending=[False, True])
    summary = output / 'summary'
    summary.mkdir(exist_ok=True)
    table.write_csv(summary / 'daily.csv')
    totals.write_csv(summary / 'strategies.csv')
    positives = totals.filter((pl.col('split') == 'val_select') & (pl.col('net_pnl_usdt') > 0))
    lines = [
        '# 宽价差原生做市结果｜2026-11-11', '',
        '使用Maker 2bps、Taker 5bps，原生保守队列，盘口可见延迟20ms，提交延迟1ms。',
        description,
        '每个交易日独立从10000USDT空仓开始并实际平仓，下表为各独立日账户损益合计。', '',
        '| 分区 | 规则 | 日期数 | 成交数 | 费用前USDT | 费用USDT | 净收益USDT | 剔除最大盈利日USDT |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in totals.iter_rows(named=True):
        lines.append(f"| {row['split']} | {row['variant_id']} | {row['days']} | {row['fills']} | {row['gross_pnl_usdt']:.3f} | {row['fees_usdt']:.3f} | {row['net_pnl_usdt']:.3f} | {row['excluding_best_day_usdt']:.3f} |")
    (summary / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json_atomic(output / 'completed.json', {'status': 'success', 'positive_native_candidates': positives.to_dicts()})
