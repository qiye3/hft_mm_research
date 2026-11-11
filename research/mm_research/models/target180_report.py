"""汇总180秒GPU基线，保留每个种子，并按交易日配对比较同列序列模型。"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic


# 【核心逻辑312】从已完成产物汇总预测能力，日级重采样比较同种子的序列与当前时点MLP。
def summarize_models(mlp_root: Path, sequence_root: Path) -> None:
    """只读取指标和日表，失败实验单列，避免再次加载完整预测矩阵。"""
    records = []
    incomplete = []
    sources = [
        (mlp_root, 'mlp_lr*'),
        (sequence_root, 'mlp_current_seed*'),
        (sequence_root, '*_120s_seed*'),
        (mlp_root, 'xgb_*'),
    ]
    for root, pattern in sources:
        for folder in sorted(root.glob(pattern)):
            if not (folder / 'completed.json').exists():
                incomplete.append(str(folder))
                continue
            metrics = json.loads((folder / 'metrics.json').read_text())
            resources = json.loads((folder / 'resource_summary.json').read_text())
            progress = json.loads((folder / 'progress.json').read_text())
            for split, values in metrics.items():
                records.append({
                    'model': folder.name, 'split': split, **values['all'],
                    'nonoverlap_phase0_ic': values['phases']['0']['ic'],
                    'nonoverlap_phase60_ic': values['phases']['60']['ic'],
                    'nonoverlap_phase120_ic': values['phases']['120']['ic'],
                    'best_epoch': progress['best_epoch'],
                    'elapsed_seconds': resources['elapsed_seconds'],
                    'peak_gpu_allocated_gib': resources['peak_gpu_allocated_gib'],
                })
    frame = pl.DataFrame(records)
    frame.write_csv(mlp_root / 'model_summary.csv')
    paired = []
    for family in ['gru', 'tcn']:
        for seed in [17, 42, 73]:
            candidate = sequence_root / f'{family}_120s_seed{seed}'
            baseline = sequence_root / f'mlp_current_seed{seed}'
            if not ((candidate / 'completed.json').exists() and (baseline / 'completed.json').exists()):
                continue
            for split in ['val_dev', 'val_select']:
                daily = pl.read_parquet(candidate / f'{split}_daily.parquet').join(
                    pl.read_parquet(baseline / f'{split}_daily.parquet'), on='utc_day', suffix='_baseline')
                differences = daily.select((pl.col('ic') - pl.col('ic_baseline')).alias('difference'))
                values = differences.drop_nulls()['difference'].to_numpy()
                generator = np.random.default_rng(180)
                samples = generator.integers(0, len(values), size=(2000, len(values)))
                estimates = values[samples].mean(axis=1)
                paired.append({
                    'family': family, 'seed': seed, 'split': split, 'days': len(values),
                    'mean_daily_ic_increment': float(values.mean()),
                    'ci95_low': float(np.quantile(estimates, 0.025)),
                    'ci95_high': float(np.quantile(estimates, 0.975)),
                    'positive_day_fraction': float((values > 0).mean()),
                })
    write_json_atomic(mlp_root / 'paired_daily_comparison.json', {'comparisons': paired})
    lines = [
        '# 180秒GPU模型结果', '',
        '主目标为未来180秒对数收益bps，Train截至2026-03-22，使用两段验证。', '',
        '146列MLP复用旧冻结因子。同列序列实验只用拟合前缀选出的48列，MLP读取当前时点，GRU与TCN读取完整过去120秒。', '',
        '| 模型 | 分区 | IC | Rank IC | R²相对训练均值 | 非重叠IC相位0 | 最优轮次 |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in records:
        lines.append(
            f"| {row['model']} | {row['split']} | {row['ic']:.5f} | {row['rank_ic']:.5f} "
            f"| {row['r2_vs_train_mean']:.6f} | {row['nonoverlap_phase0_ic']:.5f} | {row['best_epoch']} |")
    lines.extend(['', '## 同列序列增量', '',
                  '以下为相对同种子48列当前时点MLP的日均IC差，区间按UTC交易日配对Bootstrap计算。', '',
                  '| 模型 | 种子 | 分区 | 日均IC差 | 95%区间 | 改善日比例 |',
                  '| --- | --- | --- | --- | --- | --- |'])
    for row in paired:
        lines.append(
            f"| {row['family']} | {row['seed']} | {row['split']} "
            f"| {row['mean_daily_ic_increment']:.5f} "
            f"| [{row['ci95_low']:.5f}, {row['ci95_high']:.5f}] "
            f"| {row['positive_day_fraction']:.1%} |")
    if incomplete:
        lines.extend(['', '## 未完成实验', '', *[f'- {name}' for name in incomplete]])
    (mlp_root / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json_atomic(mlp_root / 'summary_completed.json', {'models': len(records) // 2, 'incomplete': incomplete})


# 【非核心】显式指定两批GPU结果目录，汇总不重新训练或读取历史Test。
def main() -> None:
    """输出汇总CSV、日级对照JSON及中文报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mlp_root', type=Path)
    parser.add_argument('sequence_root', type=Path)
    parser.add_argument('--ephemeral-input', type=Path)
    arguments = parser.parse_args()
    summarize_models(arguments.mlp_root, arguments.sequence_root)
    if arguments.ephemeral_input is not None:
        arguments.ephemeral_input.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
