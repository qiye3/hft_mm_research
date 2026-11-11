"""OOF校准、逐期限共同样本评价与验证期冻结选择。"""

import json
from pathlib import Path
import tempfile

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import HORIZONS, context, open_dataset, DAY


# 【非核心】读取一个模型的明确预测分区，保留完整时间位置后再为评价筛选标签。
def prediction_frame(output: Path, model: str, segment: str) -> pl.DataFrame:
    """训练OOF按前向折拼接，验证只读取对应冻结文件。"""
    folder = output / 'e1/predictions' / model
    files = sorted(folder.glob('oof_*.parquet')) if segment == 'train_oof' else [folder / f'{segment}.parquet']
    columns = ['decision_ts_ns', 'ready_ts_ns', 'row', 'pred_curve_bps']
    return pl.concat([pl.read_parquet(file, columns=columns) for file in files])


# 【核心逻辑267】用前向OOF预测拟合校准和分段误差风险，验证数据仅用于评价。
def calibrate_model(output: Path, model: str) -> dict:
    """每期限保存斜率、截距、中性区间、分组阈值和完整分段协方差。"""
    file = output / 'e1/calibration' / f'{model}.json'
    if file.exists():
        return json.loads(file.read_text())
    manifest, splits, matrix, targets, market = open_dataset(output)
    source_files = sorted((output / 'e1/predictions' / model).glob('oof_*.parquet'))
    capacity = sum(pl.scan_parquet(source).select(pl.len()).collect().item() for source in source_files)
    totals = np.zeros((16, 6))
    destination = output / 'e1/calibration'
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.oof-calibration-', dir=destination) as temporary:
        staging = Path(temporary)
        forecasts = np.lib.format.open_memmap(staging / 'predicted.npy', mode='w+', dtype='float64', shape=(capacity, 16))
        positions = np.lib.format.open_memmap(staging / 'rows.npy', mode='w+', dtype='int64', shape=(capacity,))
        count = 0
        for source in source_files:
            query = pl.scan_parquet(source).select('row', 'pred_curve_bps')
            for frame in query.collect_batches(chunk_size=65536, maintain_order=True, engine='streaming'):
                rows = frame['row'].to_numpy()
                actual = np.asarray(targets[rows])
                valid = np.isfinite(actual).all(axis=1)
                predicted = frame['pred_curve_bps'].to_numpy()[valid]
                actual = actual[valid]
                rows = rows[valid]
                forecasts[count:count + len(rows)] = predicted
                positions[count:count + len(rows)] = rows
                count += len(rows)
                for index in range(16):
                    usable = np.isfinite(predicted[:, index])
                    forecast = predicted[usable, index]
                    truth = actual[usable, index]
                    totals[index] += [len(truth), forecast.sum(), truth.sum(),
                                      forecast @ forecast, forecast @ truth, truth @ truth]
        slopes = []
        intercepts = []
        thresholds = []
        neutral = []
        for index in range(16):
            observations, sum_forecast, sum_truth, square_forecast, cross, square_truth = totals[index]
            if observations == 0:
                slopes.append(None)
                intercepts.append(None)
                thresholds.append(None)
                neutral.append(None)
                continue
            mean_forecast = sum_forecast / observations
            mean_truth = sum_truth / observations
            variance = square_forecast / observations - mean_forecast ** 2
            slope = (cross / observations - mean_forecast * mean_truth) / variance if variance > 1e-16 else 0.
            intercept = mean_truth - slope * mean_forecast
            column = np.asarray(forecasts[:count, index])
            column = column[np.isfinite(column)] * slope + intercept
            slopes.append(float(slope))
            intercepts.append(float(intercept))
            thresholds.append(np.quantile(column, [.2, .4, .6, .8]).tolist())
            neutral.append(float(np.quantile(np.abs(column), .2)))
        error_sum = np.zeros(16)
        error_cross = np.zeros((16, 16))
        error_count = 0
        for begin in range(0, count, 65536):
            end = min(begin + 65536, count)
            rows = positions[begin:end]
            predicted = np.asarray(forecasts[begin:end])
            calibrated = predicted * np.array(slopes, dtype=float) + np.array(intercepts, dtype=float)
            valid = np.isfinite(calibrated).all(axis=1)
            if not valid.any():
                continue
            residual = np.asarray(targets[rows[valid]]) - calibrated[valid]
            errors = np.diff(np.column_stack([np.zeros(len(residual)), residual]), axis=1)
            error_sum += errors.sum(axis=0)
            error_cross += errors.T @ errors
            error_count += len(errors)
        covariance = None
        if error_count > 1:
            covariance = ((error_cross - np.outer(error_sum, error_sum) / error_count) / (error_count - 1)).tolist()
    record = {'model_id': model, 'fitted_on': 'strict_forward_train_oof', 'slope': slopes,
              'intercept_bps': intercepts, 'quantile_thresholds': thresholds, 'neutral_band_bps': neutral,
              'increment_error_covariance_bps2': covariance,
              'risk_type': 'increment_residual_covariance', 'oof_rows': count}
    write_json_atomic(file, record)
    return record


# 【非核心】将保存的OOF校准参数应用于当前预测，缺少的M0期限保持NaN。
def calibrated_curve(frame: pl.DataFrame, calibration: dict) -> np.ndarray:
    """返回bps预测，参数仅来自前向Train OOF。"""
    return (frame['pred_curve_bps'].to_numpy() * np.array(calibration['slope'], dtype=float)
            + np.array(calibration['intercept_bps'], dtype=float))


# 【非核心】计算相关、训练均值基准R²和校准斜率，常量输入的相关记为空。
def score_curve(actual: np.ndarray, predicted: np.ndarray, train_mean: float) -> dict:
    """数组需为同一评价行集，分母为该训练折固定的恒定均值预测误差。"""
    variance = np.var(predicted)
    slope = float(np.mean((actual - actual.mean()) * (predicted - predicted.mean())) / variance) if variance > 1e-16 else None
    correlation = float(np.corrcoef(actual, predicted)[0, 1]) if variance > 1e-16 else None
    error = actual - predicted
    baseline = np.mean((actual - train_mean) ** 2)
    return {'n_obs': len(actual), 'ic': correlation,
            'spearman_ic': float(spearmanr(actual, predicted).statistic) if variance > 1e-16 else None,
            'mse_bps2': float(np.mean(error ** 2)),
            'oos_r2': float(1 - np.mean(error ** 2) / baseline) if baseline > 0 else None,
            'calibration_slope': slope, 'prediction_std_bps': float(np.std(predicted)),
            'mean_realized_bps': float(actual.mean()), 'mean_predicted_bps': float(predicted.mean())}


# 【核心逻辑268】所有模型在同一16期限共同有效行评价，选择只读取val_select。
def evaluate_models(filename: Path) -> None:
    """输出每期限指标和日IC，最多两个后续模型；M0仅报告实际存在的三个期限。"""
    root, config, output = context(filename)
    manifest, splits, matrix, targets, market = open_dataset(output)
    folder = output / 'e1'
    records = []
    daily = []
    registry = []
    ranking = []
    for model in ['M0', 'M1', 'M2', 'M3', 'M4']:
        complete = all((folder / 'models' / model / fold / 'completed.json').exists()
                       for fold in ['oof_0', 'oof_1', 'oof_2', 'final'])
        if not complete:
            continue
        calibration = calibrate_model(output, model)
        prep = np.load(folder / 'folds/final/preprocessing.npz')
        registry.append({'model_id': model, 'status': 'complete', 'feature_hash': manifest['feature_hash'],
                         'splits_hash': manifest['splits_hash'], 'target_unit': 'log_return_bps',
                         'prediction_clock': 'decision_ts_plus_1ms', 'test_inference_executed': False})
        model_scores = []
        for segment in ['val_dev', 'val_select']:
            frame = prediction_frame(output, model, segment)
            rows = frame['row'].to_numpy()
            actual = np.asarray(targets[rows])
            predicted = calibrated_curve(frame, calibration)
            common = np.isfinite(actual).all(axis=1)
            for index, horizon in enumerate(HORIZONS):
                valid = common & np.isfinite(predicted[:, index])
                if not valid.any():
                    continue
                truth = actual[valid, index]
                forecast = predicted[valid, index]
                score = score_curve(truth, forecast, prep['target_mean'][index])
                records.append({'model_id': model, 'split': segment, 'horizon_s': int(horizon), **score})
                if segment == 'val_select' and model != 'M0':
                    model_scores.append(score['mse_bps2'] / prep['target_scale'][index] ** 2)
                days = rows[valid] // DAY
                for day in np.unique(days):
                    mask = days == day
                    if mask.sum() < 100:
                        continue
                    variance = np.var(forecast[mask])
                    daily.append({'model_id': model, 'split': segment, 'horizon_s': int(horizon),
                                  'date': manifest['dates'][int(day)], 'n_obs': int(mask.sum()),
                                  'ic': float(np.corrcoef(truth[mask], forecast[mask])[0, 1]) if variance > 1e-16 else None,
                                  'mse_bps2': float(np.mean((truth[mask] - forecast[mask]) ** 2))})
        if model_scores:
            ranking.append({'model_id': model, 'normalized_mse': float(np.mean(model_scores))})
    if not records:
        raise ValueError('没有完成前向OOF和最终训练的模型')
    pl.DataFrame(records).write_csv(folder / 'metrics_by_horizon.csv')
    pl.DataFrame(daily).write_parquet(folder / 'daily_metrics.parquet')
    ranking.sort(key=lambda row: row['normalized_mse'])
    alternatives = [row['model_id'] for row in ranking if row['model_id'] != 'M1']
    selected = ['M1'] + alternatives[:1]
    write_json_atomic(folder / 'model_registry.json', registry)
    write_json_atomic(folder / 'selected_models.json', {'models': selected, 'ranking': ranking,
                      'selection_split': 'val_select', 'test_used': False})
    lines = ['# E1多期限模型对照｜2026-11-09', '',
             '全部模型共用固定因子、60/20/20切分和16期限标签有效样本。校准只用前向OOF。', '',
             '| 模型 | 分区 | 期限s | IC | R² | 预测标准差bps |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for record in records:
        if record['horizon_s'] in [1, 10, 30, 60, 120, 300]:
            lines.append(f'| {record["model_id"]} | {record["split"]} | {record["horizon_s"]} | '
                         f'{record["ic"]} | {record["oos_r2"]} | {record["prediction_std_bps"]:.5f} |')
    (folder / 'model_comparison.md').write_text('\n'.join(lines) + '\n')
    print('E1_EVALUATION_READY', selected, flush=True)
