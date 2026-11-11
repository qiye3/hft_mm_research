"""核对四档因子完整性，汇总七周期评价并发布180秒训练期窗口选择结果。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import yaml

from mm_research.data.storage import read_metadata, write_json_atomic
from mm_research.factors.dataset import factor_module


# 【非核心】读取单因子页脚核对全日期网格、时间端点、Float64及生产定义。
def inspect_factor(task: tuple) -> dict:
    """返回可复用文件清单，缺列、少行或日期不完整时明确终止验收。"""
    folder, batch, name, dates = task
    file = folder / f'{name}.parquet'
    metadata = read_metadata(file)
    parquet = pq.ParquetFile(file)
    if metadata['definition']['dates'] != dates or parquet.metadata.num_rows != len(dates) * 86400:
        raise ValueError(f'因子全日期行数或定义不完整 {name}')
    if pl.read_parquet_schema(file)[name] != pl.Float64:
        raise ValueError(f'因子类型不为Float64 {name}')
    time_index = parquet.schema_arrow.get_field_index('timestamp_ns')
    first = parquet.metadata.row_group(0).column(time_index).statistics.min
    last = parquet.metadata.row_group(parquet.metadata.num_row_groups - 1).column(time_index).statistics.max
    expected_first = int(dt.datetime.fromisoformat(dates[0]).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    expected_last = int(dt.datetime.fromisoformat(dates[-1]).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000 + 86399_000_000_000
    if first != expected_first or last != expected_last:
        raise ValueError(f'因子首尾时间不完整 {name}')
    return {'batch': batch, 'factor': name, 'rows': parquet.metadata.num_rows,
            'bytes': file.stat().st_size, 'reused': file.is_symlink(), 'source_file': str(file.resolve())}


# 【非核心】按180秒训练期共同样本Rank IC为每个基础公式选择一个聚合窗口。
def select_windows(summary: pl.DataFrame, daily_ic: pl.DataFrame, train_dates: list[str]) -> tuple:
    """三折方向只在过去前缀确定，候选各折覆盖至少80%，以未来折定向Rank IC中位数选窗。"""
    training_days = len(train_dates)
    edges = [int(training_days * fraction) for fraction in [.4, .6, .8, 1.]]
    fold_scores = []
    for index in range(3):
        first, last = edges[index], edges[index + 1]
        prefix = daily_ic.filter(pl.col('date') < train_dates[first]).group_by('factor').agg(
            pl.col('rank_ic').mean().sign().alias('direction'), pl.col('rank_ic').count().alias('prefix_days'))
        forward = daily_ic.filter(pl.col('date').is_in(train_dates[first:last])).group_by('factor').agg(
            pl.col('rank_ic').mean().alias('forward_raw_rank_ic'), pl.col('rank_ic').count().alias('forward_days'))
        scores = forward.join(prefix, on='factor').with_columns(
            (pl.col('forward_raw_rank_ic') * pl.col('direction')).alias('forward_rank_ic'),
            pl.lit(index).alias('fold'))
        scores = scores.filter(pl.col('forward_rank_ic').is_finite() & (pl.col('direction').abs() == 1)
                               & (pl.col('prefix_days') >= int(np.ceil(first * .8)))
                               & (pl.col('forward_days') >= int(np.ceil((last - first) * .8))))
        fold_scores.append(scores)
    folds = pl.concat(fold_scores)
    scores = folds.group_by('factor').agg(
        pl.len().alias('eligible_folds'), pl.col('direction').n_unique().alias('direction_variants'),
        pl.col('forward_rank_ic').median().alias('forward_rank_ic'))
    scores = scores.filter((pl.col('eligible_folds') == 3) & (pl.col('direction_variants') == 1)
                           & (pl.col('forward_rank_ic') > 0))
    training = summary.filter((pl.col('split') == 'train') & (pl.col('sampling') == 'common')
                              & (pl.col('horizon_ns') == 180_000_000_000)
                              & pl.col('rank_ic_mean').is_finite()
                              & (pl.col('valid_ic_days') >= int(np.ceil(training_days * .8))))
    training = training.join(scores, on='factor').with_columns(
        pl.col('factor').str.replace(r'_(?:\d+s_)?\d+s$', '').alias('formula'),
        pl.col('factor').str.extract(r'_(\d+s(?:_\d+s)?)$', 1).alias('window'),
        pl.col('factor').str.extract(r'_(\d+)s$', 1).cast(pl.Int64).alias('long_window_s'),
        pl.col('rank_ic_mean').abs().alias('absolute_rank_ic'))
    selected = training.sort(['formula', 'forward_rank_ic', 'long_window_s'], descending=[False, True, False]).unique(
        subset=['formula'], keep='first', maintain_order=True).sort('forward_rank_ic', descending=True)
    return selected, folds


# 【核心逻辑293】全部35组验收后发布完整清单、窗口选择及七周期汇总图。
def finalize_windows(config_file: Path, evaluation_root: Path, output: Path) -> None:
    """只使用已完成因子与评价结果，所有文件齐备后原子发布完成标记。"""
    config = yaml.safe_load(config_file.read_text())
    folder = Path(config['factor_root']) / config['factor_version']
    tasks = []
    for batch in config['batches']:
        names = factor_module(batch).factor_expressions(config)
        tasks.extend((folder, batch, name, config['dates']) for name in names)
    with ThreadPoolExecutor(max_workers=8) as workers:
        inventory = pl.DataFrame(list(workers.map(inspect_factor, tasks)))
    if inventory['factor'].n_unique() != 1206:
        raise ValueError('四档因子数量不为预期1206个')
    frames = []
    for batch in config['batches']:
        summary_file = evaluation_root / batch / 'summary.parquet'
        frame = pl.read_parquet(summary_file)
        frames.append(frame.with_columns(pl.lit(batch).alias('batch')))
    summary = pl.concat(frames, how='diagonal_relaxed')
    if set(summary['factor'].unique()) != set(inventory['factor']):
        raise ValueError('已评价因子与已生产清单不一致')
    train_dates = [date for date in config['dates'] if date <= '2026-03-22']
    daily_files = [evaluation_root / row['batch'] / row['factor'] / 'daily_ic.parquet'
                   for row in inventory.iter_rows(named=True)]
    daily_ic = pl.scan_parquet(daily_files).filter(
        (pl.col('horizon_ns') == 180_000_000_000) & (pl.col('sampling') == 'common')
        & pl.col('date').is_in(train_dates)).select('factor', 'date', 'rank_ic').collect(engine='streaming')
    selected, folds = select_windows(summary, daily_ic, train_dates)
    output.mkdir(parents=True, exist_ok=True)
    inventory.write_csv(output / 'factor_inventory.csv')
    summary.write_parquet(output / 'summary.parquet', compression='zstd')
    selected.write_csv(output / 'selected_windows_train_180s.csv')
    folds.write_csv(output / 'window_selection_forward_folds.csv')
    selected_metrics = summary.filter(pl.col('factor').is_in(selected['factor']) & (pl.col('sampling') == 'common'))
    selected_metrics.write_csv(output / 'selected_windows_multihorizon.csv')
    overview = summary.filter(pl.col('sampling') == 'common').with_columns(
        pl.col('factor').str.extract(r'_(\d+s(?:_\d+s)?)$', 1).alias('window'))
    overview = overview.group_by('split', 'window', 'horizon_ns').agg(
        pl.col('rank_ic_mean').abs().median().alias('median_absolute_rank_ic'),
        pl.col('factor').n_unique().alias('factors'))
    overview.write_csv(output / 'window_horizon_overview.csv')
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    horizons = [1, 10, 30, 60, 120, 180, 300]
    windows = ['10s', '60s', '180s', '600s']
    for axis, split in zip(axes, ['train', 'val_dev', 'val_select']):
        cells = overview.filter(pl.col('split') == split)
        values = np.full((len(windows), len(horizons)), np.nan)
        for row in cells.iter_rows(named=True):
            if row['window'] in windows:
                values[windows.index(row['window']), horizons.index(row['horizon_ns'] // 1_000_000_000)] = row['median_absolute_rank_ic']
        artist = axis.imshow(values, aspect='auto', cmap='viridis')
        axis.set_xticks(range(len(horizons)), horizons)
        axis.set_yticks(range(len(windows)), windows)
        axis.set(title=split, xlabel='Prediction horizon, seconds', ylabel='Aggregation window')
        figure.colorbar(artist, ax=axis, label='Median absolute daily Rank IC')
    figure.savefig(output / 'window_horizon_overview.png', dpi=160)
    plt.close(figure)
    lines = ['# 四档因子与多周期评价｜2026-11-10', '',
             f'已验收{inventory.height}个全日期因子，每个文件{len(config["dates"]) * 86400:,}行。',
             f'其中{inventory["reused"].sum()}个文件复用相同定义产物。',
             '聚合窗口为10、60、180、600秒，主目标180秒。',
             '窗口选择使用Train内三折前向共同样本，各折方向只由过去前缀确定。历史Test列为回看复核。', '',
             f'通过训练覆盖筛选的基础公式共{selected.height}个。', '',
             '| 因子 | 前向定向Rank IC中位数 | 训练日均Rank IC |', '| --- | ---: | ---: |']
    for row in selected.head(30).iter_rows(named=True):
        lines.append(f'| {row["factor"]} | {row["forward_rank_ic"]:.6f} | {row["rank_ic_mean"]:.6f} |')
    lines.extend(['', '完整指标见summary.parquet，窗口选择见selected_windows_train_180s.csv。',
                  '热力图见window_horizon_overview.png，各组目录保存逐日IC、分组收益、OLS和180秒后段收益。', ''])
    (output / 'report.md').write_text('\n'.join(lines))
    write_json_atomic(output / 'completed.json', {'factors': inventory.height, 'reused': int(inventory['reused'].sum()),
                      'rows_per_factor': len(config['dates']) * 86400, 'selected_formulas': selected.height})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    finalize_windows(arguments.config, arguments.evaluation_root, arguments.output)
