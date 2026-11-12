"""现货信息消融和三种子组合，直接复用首轮缓存与已完成的seed17。"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import context, SECOND, DAY


# 【核心逻辑351】显式链接已有变量和模型，仅为未运行的组合安排训练。
def prepare_ablation(filename: Path) -> None:
    """首轮产物保持不变，本轮补两个基线种子、两个现货种子和六个消融模型。"""
    _, config, output = context(filename)
    previous = Path(config['previous'])
    links = {
        'cross.npy': previous / 'cross.npy',
        'features_ready.json': previous / 'features_ready.json',
        'spot/inputs': previous / 'spot/inputs',
        'spot/models/180s_combined_xgb_seed17': previous / 'spot/models/180s_combined_xgb_seed17',
        'baseline/models/180s_old_xgb_seed17': Path(config['baseline']) / '180s_old_xgb_seed17',
    }
    for name, source in links.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    write_json_atomic(output / 'prepared.json', {
        'status': 'success', 'reuse': {name: str(path) for name, path in links.items()},
    })


# 【核心逻辑352】逐行平均三个种子的预测，不用三个IC的均值冒充组合IC。
def read_ensemble(output: Path, group: str, split: str) -> pl.DataFrame:
    """保留全部决策秒，标签有效性只在评价时参与筛选。"""
    input_group = 'old' if group == 'baseline' else 'combined'
    folder = output / group / 'models'
    table = pl.read_parquet(folder / f'180s_{input_group}_xgb_seed17/{split}.parquet').rename({
        'prediction_log_bps': 'prediction_17',
    })
    for seed in [42, 73]:
        prediction = pl.read_parquet(
            folder / f'180s_{input_group}_xgb_seed{seed}/{split}.parquet',
            columns=['timestamp_ns', 'prediction_log_bps'],
        ).rename({'prediction_log_bps': f'prediction_{seed}'})
        table = table.join(prediction, on='timestamp_ns', how='left', validate='1:1')
    return table.select(
        'timestamp_ns', 'target_log_bps', 'evaluation_valid',
        ((pl.col('prediction_17') + pl.col('prediction_42') + pl.col('prediction_73')) / 3).alias('prediction_log_bps'),
    )


# 【非核心】按连续七日块重采样配对差，不把重叠秒标签视为独立观测。
def block_interval(differences: np.ndarray) -> tuple[float, float]:
    """返回1000次循环七日块重采样的均值95%区间。"""
    random = np.random.default_rng(42)
    count = len(differences)
    starts = random.integers(0, count, size=(1000, (count + 6) // 7))
    positions = ((starts[:, :, None] + np.arange(7)) % count).reshape(1000, -1)[:, :count]
    averages = differences[positions].mean(axis=1)
    return tuple(float(value) for value in np.quantile(averages, [0.025, 0.975]))


# 【非核心】记录各随机种子对同种子基线的差，避免只展示表现最好的种子。
def seed_metrics(output: Path, group: str, split: str) -> list[dict]:
    """读取已保存评分，不重复生成预测。"""
    records = []
    for seed in [17, 42, 73]:
        current_file = output / group / f'models/180s_combined_xgb_seed{seed}/metrics.json'
        old_file = output / 'baseline' / f'models/180s_old_xgb_seed{seed}/metrics.json'
        current = json.loads(current_file.read_text())[split]['all']
        baseline = json.loads(old_file.read_text())[split]['all']
        records.append({
            'group': group, 'seed': seed, 'split': split, 'ic': current['ic'],
            'baseline_ic': baseline['ic'], 'delta_ic': current['ic'] - baseline['ic'],
        })
    return records


# 【核心逻辑353】比较真实三种子组合与同样本基线，保存消融结果和可复用的预测信号。
def report_ablation(filename: Path) -> None:
    """输出总体指标、逐日增量与七日块区间，不按历史复核结果重新选择种子。"""
    _, config, output = context(filename)
    summary = output / 'summary'
    summary.mkdir(exist_ok=True)
    signals = output / 'signals'
    signals.mkdir(exist_ok=True)
    records = []
    seeds = []
    for split in ['val_dev', 'val_select']:
        baseline = read_ensemble(output, 'baseline', split).rename({'prediction_log_bps': 'baseline_prediction'})
        for group in ['flow', 'price', 'spot']:
            prediction = read_ensemble(output, group, split)
            prediction.write_parquet(signals / f'{group}_{split}.parquet', compression='zstd')
            paired = prediction.join(
                baseline.select('timestamp_ns', 'baseline_prediction'), on='timestamp_ns', validate='1:1',
            ).filter(pl.col('evaluation_valid'))
            actual = pl.col('target_log_bps')
            current = pl.col('prediction_log_bps')
            old = pl.col('baseline_prediction')
            metrics = paired.select(
                pl.len().alias('rows'), pl.corr(current, actual).alias('ic'),
                pl.corr(old, actual).alias('baseline_ic'),
                pl.corr(current, actual, method='spearman').alias('rank_ic'),
                (current - actual).pow(2).mean().alias('mse'),
                (old - actual).pow(2).mean().alias('baseline_mse'),
            ).row(0, named=True)
            daily = paired.with_columns((pl.col('timestamp_ns') // (SECOND * DAY)).alias('utc_day')).group_by('utc_day').agg(
                pl.corr(current, actual).alias('ic'), pl.corr(old, actual).alias('baseline_ic'),
            ).sort('utc_day').with_columns((pl.col('ic') - pl.col('baseline_ic')).alias('delta_ic'))
            daily.write_csv(summary / f'{group}_{split}_daily.csv')
            differences = daily.filter(pl.col('delta_ic').is_finite())['delta_ic'].to_numpy()
            lower, upper = block_interval(differences)
            records.append({
                'group': group, 'split': split, **metrics,
                'mean_daily_delta_ic': float(differences.mean()),
                'positive_day_share': float((differences > 0).mean()),
                'block7_ci95_low': lower, 'block7_ci95_high': upper,
            })
            seeds.extend(seed_metrics(output, group, split))
    pl.DataFrame(records).write_csv(summary / 'ensembles.csv')
    pl.DataFrame(seeds).write_csv(summary / 'seeds.csv')
    lines = [
        '# 现货信息消融与三种子优化结果｜2026-11-11', '',
        '模型均为原参数XGBoost。基线和新增信息均使用17/42/73种子，主表为逐秒预测平均后的真实组合表现。',
        'flow加入4列成交压力，price加入12列现货收益及两市场价格联动，spot加入全部16列。',
        '补训10个此前未运行的模型，原seed17基线和现货模型直接复用。时间划分、180秒目标、拟合行不变。', '',
        '| 新增信息 | 分区 | 组合IC | 基线组合IC | 日均IC增量 | 连续7日块95%区间 |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for record in records:
        lines.append(
            f"| {record['group']} | {record['split']} | {record['ic']:.5f} | {record['baseline_ic']:.5f} | "
            f"{record['mean_daily_delta_ic']:.5f} | [{record['block7_ci95_low']:.5f}, {record['block7_ci95_high']:.5f}] |"
        )
    lines.extend(['', '本轮是已有历史上的固定消融研究，区间为描述性统计；三组均完整展示，不按复核成绩挑选随机种子。',
                  '主表及MSE见ensembles.csv，逐种子对照见seeds.csv，逐日指标另存CSV。组合信号保存在signals，供后续交易评估复用。'])
    (summary / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json_atomic(output / 'completed.json', {'status': 'success', 'new_models': 10, 'reused_models': 2})
    print(json.dumps(records, ensure_ascii=False, indent=2), flush=True)
