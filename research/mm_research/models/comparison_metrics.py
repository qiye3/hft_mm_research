"""同一评估行集上的分块预测、冻结分组阈值和配对日级比较。"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mm_research.models.experiment_data import HORIZONS, history_batch


# 【核心逻辑142】使用同一缺失处理生成预测，GPU和CPU模型接收相同的当前列。
def predict_comparison(record: dict, model, matrix, rows: np.ndarray, metadata: dict) -> np.ndarray:
    """神经序列只扩展过去观察范围，预测输出全部还原为bps。"""
    columns = np.asarray(record['columns'], dtype=np.int64)
    inputs = history_batch(matrix, rows, columns, record['sequence_seconds'])
    if record['family'] == 'gru':
        import torch
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            values = model(torch.from_numpy(inputs).to('cuda')).float().cpu().numpy()
        return values.astype(np.float64) * metadata['target_scale_bps'][0]
    if record['family'] == 'lightgbm':
        return model.predict(inputs[:, -1], num_threads=2)
    return inputs[:, -1] @ np.asarray(model['coefficients']) + model['intercept']


# 【非核心】把共同有效样本的可加统计量转换成统一的预测指标。
def prediction_metrics(totals: np.ndarray) -> dict:
    """七项统计顺序为样本数、y和p的一二阶矩、交叉乘积及平方误差。"""
    count, sum_y, sum_p, sum_yy, sum_pp, sum_yp, error = totals
    variance_y = max(sum_yy - sum_y * sum_y / count, 0.)
    variance_p = max(sum_pp - sum_p * sum_p / count, 0.)
    denominator = np.sqrt(variance_y * variance_p)
    return {'rows': int(count), 'mse_bps2': float(error / count), 'rmse_bps': float(np.sqrt(error / count)),
            'ic': float((sum_yp - sum_y * sum_p / count) / denominator) if denominator else None,
            'r2': float(1 - error / variance_y) if variance_y else None}


# 【核心逻辑146】分块输出全部共同样本预测，保存五周期日级IC和训练阈值下的分组收益。
def evaluate_comparison(record: dict, model, matrix, targets, rows: np.ndarray,
                        metadata: dict, thresholds: list[float], destination: Path) -> dict:
    """按目标各自有效性评价，同一周期的各模型始终使用完全相同的时间戳。"""
    totals = np.zeros((5, 7), dtype=np.float64)
    nonoverlap = np.zeros((5, 7), dtype=np.float64)
    daily = {}
    groups = np.zeros((5, 5, 2), dtype=np.float64)
    temporary = destination.with_suffix('.partial.parquet')
    with pq.ParquetWriter(temporary, pa.schema([('timestamp_ns', pa.int64()), ('prediction_bps', pa.float64())]), compression='zstd') as writer:
        for offset in range(0, len(rows), 4096):
            indices = rows[offset:offset + 4096]
            prediction = predict_comparison(record, model, matrix, indices, metadata)
            if not np.isfinite(prediction).all():
                raise ValueError(f'模型输出非有限值 {record["id"]}')
            seconds = metadata['start_timestamp_seconds'] + indices
            writer.write_table(pa.table({'timestamp_ns': seconds * 1_000_000_000,
                                          'prediction_bps': prediction.astype(np.float64)}))
            assignment = np.searchsorted(thresholds, prediction, side='right')
            for horizon_index, horizon in enumerate(HORIZONS):
                response = np.asarray(targets[indices, horizon_index], dtype=np.float64)
                valid = np.isfinite(response)
                left, right = response[valid], prediction[valid]
                moment = np.array([len(left), left.sum(), right.sum(), left @ left, right @ right,
                                   left @ right, (left - right) @ (left - right)])
                totals[horizon_index] += moment
                independent = valid & (seconds % horizon == 0)
                nonoverlap[horizon_index] += prediction_totals(response[independent], prediction[independent])
                for day in np.unique(seconds[valid] // 86400):
                    mask = valid & (seconds // 86400 == day)
                    actual, predicted = response[mask], prediction[mask]
                    key = (int(day), horizon)
                    current = np.array([len(actual), actual.sum(), predicted.sum(), actual @ actual,
                                        predicted @ predicted, actual @ predicted, (actual - predicted) @ (actual - predicted)])
                    daily[key] = daily.get(key, np.zeros(7)) + current
                for group in range(5):
                    selected = valid & (assignment == group)
                    groups[horizon_index, group] += [int(selected.sum()), response[selected].sum()]
    temporary.replace(destination)
    daily_records = [{'day': day, 'horizon_seconds': horizon, **prediction_metrics(values)}
                     for (day, horizon), values in sorted(daily.items()) if values[0] > 1]
    return {'nonoverlap_horizons': {str(horizon): prediction_metrics(nonoverlap[index]) for index, horizon in enumerate(HORIZONS)},
            'horizons': {str(horizon): prediction_metrics(totals[index]) for index, horizon in enumerate(HORIZONS)},
            'daily': daily_records,
            'train_fixed_group_mean_bps': {
                str(horizon): [float(total / count) if count else None for count, total in groups[index]]
                for index, horizon in enumerate(HORIZONS)},
            'prediction_file': destination.name, 'thresholds_fit_on': 'fixed_training_subset'}


# 【核心逻辑148】对齐同一天的指标后重采样差值，两个模型共用同一组抽样日期。
def paired_day_comparison(candidate: list[dict], baseline: list[dict], seed: int) -> dict:
    """返回1s日均IC差和误差减少量，正值表示候选改善，区间共用日期抽样。"""
    candidate_days = {
        row['day']: row for row in candidate
        if row['horizon_seconds'] == 1 and row['ic'] is not None
    }
    baseline_days = {
        row['day']: row for row in baseline
        if row['horizon_seconds'] == 1 and row['ic'] is not None
    }
    dates = sorted(set(candidate_days) & set(baseline_days))
    ic_differences = np.array([
        candidate_days[day]['ic'] - baseline_days[day]['ic'] for day in dates
    ])
    mse_improvements = np.array([
        baseline_days[day]['mse_bps2'] - candidate_days[day]['mse_bps2'] for day in dates
    ])
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(dates), size=(2000, len(dates)))
    ic_estimates = ic_differences[samples].mean(axis=1)
    mse_estimates = mse_improvements[samples].mean(axis=1)
    return {
        'days': len(dates),
        'mean_daily_ic_difference': float(ic_differences.mean()),
        'ci95': np.quantile(ic_estimates, [.025, .975]).tolist(),
        'positive_day_fraction': float((ic_differences > 0).mean()),
        'mean_daily_mse_improvement_bps2': float(mse_improvements.mean()),
        'mse_improvement_ci95': np.quantile(mse_estimates, [.025, .975]).tolist(),
        'mse_positive_day_fraction': float((mse_improvements > 0).mean()),
        'resampling_unit': 'paired_utc_day',
    }


# 【非核心】由成对真实值与预测值生成可加统计，所有评分入口共用相同计算。
def prediction_totals(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """返回误差与相关所需七项Float64统计。"""
    return np.array([len(actual), actual.sum(), predicted.sum(), actual @ actual,
                     predicted @ predicted, actual @ predicted, (actual - predicted) @ (actual - predicted)])
