"""分离模型信号保持、实际未来收益曲线和远期条件增量。"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import HORIZONS, DAY, context, open_dataset
from mm_research.multihorizon.evaluation import prediction_frame, calibrated_curve, score_curve


# 【非核心】按UTC日聚合再重采样，标签重叠的逐秒观测不作为独立统计样本。
def day_mean_interval(values: np.ndarray, days: np.ndarray, seed: int) -> tuple:
    """均值置信区间使用日块加权，返回日数、标准误和95%区间。"""
    unique_days, inverse = np.unique(days, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    random = np.random.default_rng(seed)
    samples = random.integers(0, len(unique_days), size=(2000, len(unique_days)))
    estimates = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    lower, upper = np.quantile(estimates, [.025, .975])
    return len(unique_days), float(estimates.std()), float(lower), float(upper)


# 【核心逻辑269】仅用训练OOF固定强度分组，验证中同时观测信号保持和真实收益曲线。
def analyze_model_persistence(output: Path, model: str, config: dict) -> None:
    """未来更新的预测只用于事后相关诊断，不导出为当前策略输入。"""
    folder = output / 'e2' / model
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'completed.json').exists():
        return
    manifest, splits, matrix, targets, market = open_dataset(output)
    calibration = json.loads((output / 'e1/calibration' / f'{model}.json').read_text())
    training = prediction_frame(output, model, 'train_oof')
    train_prediction = calibrated_curve(training, calibration)
    train_rows = training['row'].to_numpy()
    validation = prediction_frame(output, model, 'val_dev')
    forecasts = calibrated_curve(validation, calibration)
    rows = validation['row'].to_numpy()
    actual = np.asarray(targets[rows])
    common = np.isfinite(actual).all(axis=1)
    autocorrelation = []
    for sample_name, positions, prediction in [('train_oof', train_rows, train_prediction), ('val_dev', rows, forecasts)]:
        for horizon in [1, 10, 60, 120]:
            index = int(np.flatnonzero(HORIZONS == horizon)[0])
            if calibration['neutral_band_bps'][index] is None:
                continue
            signal = prediction[:, index]
            neutral = calibration['neutral_band_bps'][index]
            thresholds = calibration['quantile_thresholds'][index]
            groups = np.searchsorted(thresholds, signal)
            for lag in [1, 2, 5, 10, 20, 30, 60, 120, 300]:
                pair = (positions[lag:] - positions[:-lag] == lag) & np.isfinite(signal[:-lag]) & np.isfinite(signal[lag:])
                for group in [-1, 0, 1, 2, 3, 4]:
                    valid = pair.copy()
                    if group >= 0:
                        valid &= groups[:-lag] == group
                    if valid.sum() < 100:
                        continue
                    first = signal[:-lag][valid]
                    second = signal[lag:][valid]
                    active = (np.abs(first) > neutral) & (np.abs(second) > neutral)
                    autocorrelation.append({
                        'model_id': model, 'signal_horizon_s': horizon, 'lag_s': lag,
                        'sample_segment': sample_name, 'quantile_group': group + 1,
                        'n_pairs': int(valid.sum()), 'n_active_pairs': int(active.sum()),
                        'autocorr': float(np.corrcoef(first, second)[0, 1]) if np.var(first) > 1e-16 else None,
                        'same_sign_all': float(np.mean(np.sign(first) == np.sign(second))),
                        'same_sign_rate': float(np.mean(np.sign(first[active]) == np.sign(second[active]))) if active.any() else None,
                        'neutral_band_bps': neutral, 'n_days': len(np.unique(positions[:-lag][valid] // DAY)),
                    })
    curves = []
    for alpha_horizon in [10, 60, 120]:
        alpha_index = int(np.flatnonzero(HORIZONS == alpha_horizon)[0])
        thresholds = calibration['quantile_thresholds'][alpha_index]
        if thresholds is None:
            continue
        group = np.searchsorted(thresholds, forecasts[:, alpha_index])
        for quantile in range(5):
            valid = common & (group == quantile) & np.isfinite(forecasts[:, alpha_index])
            if not valid.any():
                continue
            days = rows[valid] // DAY
            for index, horizon in enumerate(HORIZONS):
                values = actual[valid, index]
                count_days, standard_error, lower, upper = day_mean_interval(values, days, config['seed'])
                curves.append({'model_id': model, 'alpha_horizon_s': alpha_horizon,
                               'quantile_group': quantile + 1, 'future_horizon_s': int(horizon),
                               'n_obs': len(values), 'n_days': count_days,
                               'mean_realized_bps': float(values.mean()), 'median_realized_bps': float(np.median(values)),
                               'mean_predicted_bps': float(np.nanmean(forecasts[valid, index])),
                               'day_block_standard_error_bps': standard_error,
                               'ci_lower_bps': lower, 'ci_upper_bps': upper,
                               'directional_mean_bps': float(values.mean() * (-1 if quantile < 2 else 1))})
    incremental = []
    for start, end in [(1, 10), (10, 30), (30, 60), (60, 120), (120, 300)]:
        first = int(np.flatnonzero(HORIZONS == start)[0])
        last = int(np.flatnonzero(HORIZONS == end)[0])
        predicted = forecasts[:, last] - forecasts[:, first]
        valid = common & np.isfinite(predicted)
        if not valid.any():
            continue
        realized = actual[valid, last] - actual[valid, first]
        score = score_curve(realized, predicted[valid], 0.)
        count_days, _, lower, upper = day_mean_interval(realized, rows[valid] // DAY, config['seed'])
        incremental.append({'model_id': model, 'start_s': start, 'end_s': end, **score,
                            'n_days': count_days, 'ci_lower_bps': lower, 'ci_upper_bps': upper})
    meta = []
    for horizon in [120, 300]:
        target_index = int(np.flatnonzero(HORIZONS == horizon)[0])
        training_y = np.asarray(targets[train_rows, target_index])
        for group_name, input_horizons in [('short', [1, 10, 60]), ('curve', [1, 10, 60, 90, 120, 180, 240, 300])]:
            indices = [int(np.flatnonzero(HORIZONS == value)[0]) for value in input_horizons]
            fit = train_prediction[:, indices]
            valid_train = np.isfinite(fit).all(axis=1) & np.isfinite(training_y)
            if not valid_train.any():
                continue
            design = np.column_stack([np.ones(valid_train.sum()), fit[valid_train]])
            penalty = np.eye(design.shape[1]) * .001
            penalty[0, 0] = 0
            coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ training_y[valid_train])
            validate = forecasts[:, indices]
            valid = common & np.isfinite(validate).all(axis=1)
            prediction = np.column_stack([np.ones(valid.sum()), validate[valid]]) @ coefficients
            meta.append({'model_id': model, 'target_horizon_s': horizon, 'input_group': group_name,
                         'fitted_on': 'train_oof', 'evaluated_on': 'val_dev',
                         **score_curve(actual[valid, target_index], prediction, float(training_y[valid_train].mean()))})
    pl.DataFrame(autocorrelation).write_csv(folder / 'signal_autocorr.csv')
    pl.DataFrame(curves).write_csv(folder / 'conditional_curve.csv')
    pl.DataFrame(incremental).write_csv(folder / 'incremental_value.csv')
    pl.DataFrame(meta).write_csv(folder / 'long_horizon_increment.csv')
    valid = common & np.isfinite(forecasts).all(axis=1)
    shapes = {'model_id': model, 'rows': int(valid.sum()), 'p4_enabled': False,
              'p4_status': 'conditional_extension_not_fitted', 'future_signals_used_in_policy': False}
    if valid.any():
        values = forecasts[valid]
        shapes['peak_horizon_counts'] = {str(int(horizon)): int(np.sum(np.argmax(values, axis=1) == index))
                                         for index, horizon in enumerate(HORIZONS)}
        shapes['near_far_opposite_fraction'] = float(np.mean(values[:, 6] * (values[:, -1] - values[:, 12]) < 0))
    write_json_atomic(folder / 'curve_shapes.json', shapes)
    write_json_atomic(folder / 'completed.json', {'status': 'complete', 'model_id': model})


# 【核心逻辑270】汇总可复现模型的持续性研究，条件扩展单独记录不混入主策略。
def analyze_persistence(filename: Path) -> None:
    """仅读取Train OOF和val_dev，输出完整数值曲线供E3使用。"""
    root, config, output = context(filename)
    registry = json.loads((output / 'e1/model_registry.json').read_text())
    for record in registry:
        analyze_model_persistence(output, record['model_id'], config)
    folder = output / 'e2'
    write_json_atomic(folder / 'persistence_rules.json', {'enabled': False, 'rule_id': None,
                      'reason': '首轮保留持续性诊断，P4概率缩放作为条件扩展，未拟合不报告无增量'})
    (folder / 'persistence_report.md').write_text('# E2信号与收益持续性｜2026-11-09\n\n'
        '各模型目录保存Train OOF与val_dev自相关、同号率、训练阈值条件收益曲线、日块置信区间及长周期条件增量。'
        '未来更新的Alpha仅用于诊断。P4持续性概率模型未拟合，主对照采用P0—P3。\n')
