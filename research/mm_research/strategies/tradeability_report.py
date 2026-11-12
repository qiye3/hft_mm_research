"""180秒可交易性研究的月度分布、收益图和最终结果报告。"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import HORIZONS
from mm_research.strategies.forecast_research import research_context
from mm_research.strategies.tradeability import dataset_definition
from mm_research.strategies.tradeability_execution import summarize_mechanisms


# 【核心逻辑341】汇总费用可达性、实际交易和成交条件变化，明确后续资源是否值得继续投入。
def write_tradeability_report(filename: Path) -> None:
    """每项结论对应已生成的统计或账本，缺少阶段产物时明确报错。"""
    config, source, output = research_context(filename)
    summarize_prediction_distribution(filename)
    summarize_mechanisms(filename)
    tail_gate = json.loads((output / 'tail_gate.json').read_text())
    selected = json.loads((output / 'selected_candidates.json').read_text())
    screening = pl.DataFrame([
        json.loads(path.read_text()) for path in sorted((output / 'screening').glob('*.json'))
    ]).sort(['split', 'signal', 'quantile'])
    screening.write_csv(output / 'screening_summary.csv')
    tails = pl.read_parquet([path for path in sorted((output / 'tails').glob('*.parquet')) if '.partial.' not in path.name])
    monthly = tails.group_by('signal', 'split', 'month', 'quantile', 'state', 'direction').agg(
        pl.col('signal_rows').sum().alias('signal_rows'),
        pl.col('opportunity_n_180s').sum().alias('opportunities'),
        pl.col('opportunity_sum_180s_bps').sum().alias('signed_return_sum_bps'),
        pl.col('date').n_unique().alias('active_days'),
    ).with_columns(
        (pl.col('signed_return_sum_bps') / pl.col('opportunities')).alias('mean_signed_180s_bps'),
    ).sort('signal', 'split', 'month', 'quantile', 'state', 'direction')
    monthly.write_csv(output / 'monthly_signal_summary.csv')
    plot_tradeability_results(output)
    development = screening.filter(pl.col('split') == 'val_dev')
    review = screening.filter(pl.col('split') == 'val_select')
    reference = screening.filter((pl.col('signal') == 'h180_old_mlp') & (pl.col('quantile') == .99))
    reference_development = reference.filter(pl.col('split') == 'val_dev').row(0, named=True)
    reference_review = reference.filter(pl.col('split') == 'val_select').row(0, named=True)
    lines = [
        '# 180秒预测信号可交易性研究｜2026-11-11', '',
        '本轮复用三个冻结信号和原训练OOF校准，门槛为训练期绝对预测q90、q95、q99。',
        'Train截至2026-03-22。开发为03-23至05-14，复核为05-15至07-06，07-07至10-21作历史跨月诊断。',
        'Maker单侧2bps，Taker单侧5bps。已有历史区间均标为历史研究。', '',
        f'开发期{development.filter(pl.col("net_pnl_usdt") < 0).height}/{development.height}组亏损，'
        f'复核期{review.filter(pl.col("net_pnl_usdt") < 0).height}/{review.height}组亏损，'
        f'按开发期筛选保留{len(selected["candidates"])}个候选。',
        f'旧因子MLP训练q99门槛在开发期{reference_development["round_trips"]}次完整交易，'
        f'净收益{reference_development["net_pnl_usdt"]:.4f}USDT。'
        f'复核期{reference_review["round_trips"]}次，净收益{reference_review["net_pnl_usdt"]:.4f}USDT。',
        'q99表示原训练预测分布的99%分位，后续实际触发比例见tail_summary.csv，不能直接视为每月最强1%。', '',
        '![强信号尾部收益](figures/tail_returns.png)', '',
        '## 固定持有180秒的主动交易', '',
        '| 信号 | 分区 | 门槛 | 往返次数 | 费用前USDT | 手续费USDT | 净收益USDT |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for record in screening.iter_rows(named=True):
        lines.append(
            f'| {record["signal"]} | {record["split"]} | {record["quantile"]:.0%} | '
            f'{record["round_trips"]} | {record["gross_pnl_usdt"]:.4f} | '
            f'{record["fees_usdt"]:.4f} | {record["net_pnl_usdt"]:.4f} |'
        )
    lines.extend([
        '', '快照交易在到达时刻按最近已发生的前十档逐档成交，完整分区保留跨日持仓，资金费独立结算。',
        '三个预测模型各三个门槛在两段区间运行，共18个连续区间案例。', '',
        '![主动交易净收益](figures/active_pnl.png)', '',
        '## 尾部收益与成交归因', '',
        f'双边主动费用为{tail_gate["taker_roundtrip_fee_bps"]:.0f}bps，'
        f'超过该费用的开发期尾部均值候选数量见cost_feasibility.csv。',
        '旧MLP训练q99组的日期分块区间较宽，复核和后续月份需共同检查。',
        'tail_summary.csv保留多空方向、波动、价差分组和五个预测期限。tails目录保留逐日统计。',
        'mechanism_daily.csv记录六个固定历史日的真实挂单、成交条件价格变化与完整退出净收益。',
        '相同提交机会另用到达时刻快照逐档计算主动进出收益，并比较Maker成交与未成交子集。',
        '原生挂单使用保守队列，按400秒固定机会、两侧方向和三个挂价/等待组合独立回放。',
        '未成交机会保留，交易账本、手续费和最终空仓由已有原生核算检查。',
        '', '![相同机会的执行价值](figures/execution_value.png)', '',
    ])
    if not selected['candidates']:
        lines.extend([
            '', '## 后续阶段', '',
            '九组开发期主动交易均未产生可保留的正收益候选，本轮完成固定成交机制诊断后结束。',
            '按计划暂缓滚动训练、连续28日原生策略和候选压力测试，相关阶段标为成本筛选未通过。',
        ])
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json_atomic(output / 'completed.json', {
        'status': 'success', 'selected_candidates': selected['candidates'],
        'completed_branch': selected['next_stage'],
        'next_stage': 'candidate_execution' if selected['candidates'] else None,
        'gated_stages': ['monthly_retraining', 'continuous_28d_native', 'candidate_stress'] if not selected['candidates'] else [],
    })


# 【非核心】用实际逐日统计和交易账本生成三张研究图，图中单位与CSV一致。
def plot_tradeability_results(output: Path) -> None:
    """收益尾部保留日期分块区间，主动交易和挂单价值分别呈现。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = output / 'figures'
    folder.mkdir(exist_ok=True)
    signals = ['h180_old_mlp', 'h180_combined_mlp', 'h180_combined_xgb']
    names = ['Old MLP', 'Combined MLP', 'Combined XGBoost']
    splits = ['val_dev', 'val_select', 'test']
    colors = ['#2878b5', '#e18d23', '#35966f']
    tails = pl.read_csv(output / 'tail_summary.csv').filter(
        (pl.col('state') == 'all') & (pl.col('direction') == 0) & (pl.col('horizon_s') == 180),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for index, signal in enumerate(signals):
        axis = axes[index]
        for offset, split in enumerate(splits):
            rows = tails.filter((pl.col('signal') == signal) & (pl.col('split') == split)).sort('quantile')
            positions = np.arange(3) + (offset - 1) * .14
            axis.vlines(positions, rows['opportunity_ci95_low_bps'], rows['opportunity_ci95_high_bps'], color=colors[offset])
            axis.scatter(positions, rows['opportunity_mean_signed_bps'], color=colors[offset], label=split, s=24)
        axis.axhline(10, color='#a23333', linestyle='--', linewidth=1, label='Taker fees 10 bps')
        axis.axhline(0, color='#777777', linewidth=.7)
        axis.set(xticks=np.arange(3), xticklabels=['Train q90', 'Train q95', 'Train q99'],
                 title=names[index], ylabel='180s signed mid return (log bps)')
        axis.grid(axis='y', alpha=.2)
    axes[-1].legend(fontsize=8)
    figure.savefig(folder / 'tail_returns.png', dpi=160)
    plt.close(figure)
    screening = pl.read_csv(output / 'screening_summary.csv')
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for index, signal in enumerate(signals):
        for offset, split in enumerate(splits[:2]):
            rows = screening.filter((pl.col('signal') == signal) & (pl.col('split') == split)).sort('quantile')
            axes[index].bar(np.arange(3) + (offset - .5) * .3, rows['net_pnl_usdt'], width=.28,
                            label=split, color=colors[offset])
        axes[index].set(xticks=np.arange(3), xticklabels=['Train q90', 'Train q95', 'Train q99'],
                        title=names[index], ylabel='Net PnL (USDT)')
        axes[index].axhline(0, color='#777777', linewidth=.7)
        axes[index].grid(axis='y', alpha=.2)
    axes[-1].legend(fontsize=8)
    figure.savefig(folder / 'active_pnl.png', dpi=160)
    plt.close(figure)
    mechanism = pl.read_csv(output / 'mechanism_summary.csv').filter(pl.col('signal_group') == 'same_direction')
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for index, split in enumerate(splits[:2]):
        rows = mechanism.filter(pl.col('split') == split).sort('offset_ticks', 'ttl_ms')
        positions = np.arange(rows.height)
        axes[index].bar(positions - .15, rows['active_net_return_bps'], width=.28, label='Active / opportunity', color=colors[0])
        axes[index].bar(positions + .15, rows['maker_value_on_common_opportunities_bps'], width=.28, label='Maker / opportunity', color=colors[1])
        labels = [f'{row["offset_ticks"]} ticks / {row["ttl_ms"]} ms' for row in rows.iter_rows(named=True)]
        axes[index].set(xticks=positions, xticklabels=labels, title=split, ylabel='Net value per opportunity (bps)')
        axes[index].axhline(0, color='#777777', linewidth=.7)
        axes[index].grid(axis='y', alpha=.2)
    axes[-1].legend(fontsize=8)
    figure.savefig(folder / 'execution_value.png', dpi=160)
    plt.close(figure)


# 【核心逻辑342】按月核对预测分布与实际标签的关系，解释固定训练门槛触发率的变化。
def summarize_prediction_distribution(filename: Path) -> None:
    """只做诊断，原始预测、幅度校准、方向和触发门槛均不由这些结果修改。"""
    config, source, output = research_context(filename)
    manifest, segments = dataset_definition(source)
    targets = np.load(source / 'e0/y.npy', mmap_mode='r')
    target_index = int(np.flatnonzero(HORIZONS == 180)[0])
    records = []
    for signal in config['signals']:
        for split in ['val_dev', 'val_select', 'test']:
            bounds = segments[split]
            frame = pl.read_parquet(output / 'signals' / f'{signal["id"]}_{split}.parquet').with_columns(
                pl.Series('target_bps', targets[bounds['start_row']:bounds['stop_row'], target_index]),
                pl.from_epoch('timestamp_ns', time_unit='ns').dt.strftime('%Y-%m').alias('month'),
            )
            finite = frame.filter(pl.col('evaluation_valid') & pl.col('target_bps').is_finite())
            metrics = finite.group_by('month').agg(
                pl.len().alias('evaluation_rows'),
                pl.corr('prediction_bps', 'target_bps').alias('ic'),
                pl.corr('prediction_bps', 'target_bps', method='spearman').alias('rank_ic'),
                pl.col('prediction_log_bps').mean().alias('raw_prediction_mean_bps'),
                pl.col('prediction_log_bps').std().alias('raw_prediction_std_bps'),
                pl.col('prediction_bps').mean().alias('calibrated_prediction_mean_bps'),
                pl.col('prediction_bps').std().alias('calibrated_prediction_std_bps'),
                pl.col('target_bps').mean().alias('target_mean_bps'),
                pl.col('target_bps').std().alias('target_std_bps'),
            ).with_columns(pl.lit(signal['id']).alias('signal'), pl.lit(split).alias('split'))
            records.append(metrics)
    pl.concat(records).sort('signal', 'split', 'month').write_csv(output / 'monthly_prediction_distribution.csv')
