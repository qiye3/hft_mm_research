"""从冻结权重导出完整整秒信号，推理行集只由时间决定，标签不参与信号可用性。"""

import gc
import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.models.experiment_data import history_batch, HORIZONS

SECOND = 1_000_000_000
FACTOR_VERSIONS = ['server_factors_1s_v1', 'server_factors_extra_1s_v1',
                   'server_factors_more_1s_v1', 'server_factors_latest_1s_v1']
OBI = 'book_obi_l1_mean_3s'
DEPTH = 'book_net_depth_l10_delta_100ms_sum_5s'


# 【核心逻辑228】按已经冻结的验证选择登记信号，正在训练的权重不进入本次清单。
def freeze_signal_pool(root: Path, output: Path) -> list[dict]:
    """冻结两个夜间批次各五个目标赢家、线性基线及代表性条件模型。"""
    models = root / 'workspace/results/models'
    records = []
    references = json.loads((output / 'reference_models.json').read_text())
    for name, model in references.items():
        records.append({'id': name, 'backend': 'raw_linear', 'horizon_s': 1,
                        'columns': model['columns'], 'model': model, 'imputation': {}})
    for batch, matrix in [('overnight_legacy_v1', 'legacy_1s_v1'), ('overnight_expanded_v1', 'expanded_1s_v1')]:
        frozen = json.loads((models / batch / 'frozen_selection.json').read_text())
        for horizon, report in frozen.items():
            records.append({'id': f'{batch}_{horizon}s', 'backend': 'overnight', 'horizon_s': int(horizon),
                            'matrix': f'workspace/data/model_matrices/{matrix}', 'report': report})
    linear = json.loads((models / 'linear_1s_v1/models.json').read_text())
    for name, model in linear.items():
        records.append({'id': f'linear_{name}_1s', 'backend': 'raw_linear', 'horizon_s': 1,
                        'columns': model['columns'], 'model': model, 'imputation': {}})
    validation = root / 'workspace/results/information_validation'
    for experiment, case in [('C/v1', 'tree'), ('A2/v2', 'tree')]:
        capacity = validation / experiment / 'capacity'
        contract = json.loads((capacity / 'contract.json').read_text())
        imputation = {}
        if experiment.startswith('A2'):
            preprocessing = json.loads((validation / 'A2/v2/imputation.json').read_text())
            imputation = dict(zip(preprocessing['columns'], preprocessing['train_means']))
        records.append({'id': f'{experiment.replace("/", "_")}_{case}_1s', 'backend': 'raw_tree',
                        'horizon_s': 1, 'columns': contract['columns'][case],
                        'weights': str(capacity / f'{case}.txt'), 'imputation': imputation})
    comparison = models / 'comparison_2026-11-09_v2'
    available = json.loads((comparison / 'linear_records.json').read_text())
    for path in sorted(comparison.glob('*.json')):
        if path.stem.startswith(('lightgbm_', 'gru_')):
            available.append(json.loads(path.read_text()))
    selected = min(available, key=lambda case: case['validation_tune']['mse_bps2'])
    records.append({'id': f'comparison_{selected["id"]}', 'backend': 'comparison', 'horizon_s': 1,
                    'matrix': 'workspace/data/model_matrices/comparison_2026-11-09_v2',
                    'weights': str(comparison / selected['model_file']), 'record': selected})
    (output / 'signal_pool_frozen.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    return records


# 【核心逻辑229】按明确列名投影正式单列文件，稀疏训练位置与整日推理位置共用时间选择。
def read_raw_signal_inputs(root: Path, columns: list[str], timestamps: np.ndarray,
                           imputation: dict) -> np.ndarray:
    """返回模型原顺序矩阵，当前源缺失保留或按模型已经声明的训练均值填补。"""
    source_names = list(dict.fromkeys(name.removesuffix('__missing') for name in columns))
    values = {}
    for name in source_names:
        if name.startswith('obi_x_'):
            raise ValueError('交互消融模型需显式部署其公式，本轮使用已冻结原始输入模型')
        files = [root / f'workspace/data/features/server_531d_v1/{name}.parquet']
        files += [root / f'workspace/data/factors/{version}/{name}.parquet' for version in FACTOR_VERSIONS]
        source = next(path for path in files if path.exists())
        table = pl.scan_parquet(source).filter(pl.col('timestamp_ns').is_in(timestamps)).select('timestamp_ns', name).collect(engine='streaming')
        np.testing.assert_array_equal(table['timestamp_ns'].to_numpy(), timestamps)
        values[name] = table[name].to_numpy().copy()
    missing = {name: ~np.isfinite(value) for name, value in values.items()}
    for name, mean in imputation.items():
        if name in values:
            values[name][missing[name]] = mean
    return np.column_stack([missing[name.removesuffix('__missing')].astype(np.float64)
                            if name.endswith('__missing') else values[name] for name in columns])


# 【核心逻辑230】恢复原训练缓存与权重，整日推理完全绕过目标标签及其有效性掩码。
def predict_cached_model(root: Path, signal: dict, timestamps: np.ndarray) -> np.ndarray:
    """复用训练时的历史窗口和缺失通道，按128个决策秒分块推理。"""
    import lightgbm as lgb
    import torch
    from mm_research.models.neural import FactorNetwork
    folder = root / signal['matrix']
    metadata = json.loads((folder / 'metadata.json').read_text())
    matrix = np.load(folder / 'x.npy', mmap_mode='r')
    rows = timestamps // SECOND - metadata['start_timestamp_seconds']
    if signal['backend'] == 'overnight':
        report = signal['report']
        trial = report['trial']
        family = trial['family']
        columns = np.asarray(report['columns'], dtype=np.int64)
        sequence = trial['sequence_seconds']
        directory = root / report['directory']
        weights = directory / ('best.txt' if family == 'lightgbm' else 'best.pt')
    else:
        trial = signal['record']
        family = trial['family']
        columns = np.asarray(trial['columns'], dtype=np.int64)
        sequence = trial['sequence_seconds']
        weights = Path(signal['weights'])
    if family == 'lightgbm':
        model = lgb.Booster(model_file=str(weights))
    elif family in ['ols', 'ridge', 'lasso']:
        model = json.loads(weights.read_text())
    else:
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.15)
        model = FactorNetwork(family, len(columns) * 2, sequence, trial['hidden']).cuda().eval()
        checkpoint = torch.load(weights, weights_only=True)
        state = checkpoint['state_dict'] if signal['backend'] == 'overnight' else checkpoint
        model.load_state_dict(state)
    prediction = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(rows), 128):
        selected = rows[start:start + 128]
        history = history_batch(matrix, selected, columns, sequence)
        if family == 'lightgbm':
            result = model.predict(history[:, -1], num_threads=2)
        elif family in ['ols', 'ridge', 'lasso']:
            result = history[:, -1] @ np.asarray(model['coefficients']) + model['intercept']
        else:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                result = model(torch.from_numpy(history).cuda()).float().cpu().numpy()
            result = result * metadata['target_scale_bps'][HORIZONS.index(signal['horizon_s'])]
        prediction[start:start + len(selected)] = result
    del model, matrix
    gc.collect()
    if family not in ['lightgbm', 'ols', 'ridge', 'lasso']:
        torch.cuda.empty_cache()
    return prediction


# 【核心逻辑231】当前源无效的原始线性模型保持信号无效，训练均值填补仅用于已有声明的模型。
def predict_raw_model(root: Path, signal: dict, timestamps: np.ndarray) -> np.ndarray:
    """权重与原始列单位一致，返回含明确当前缺失的完整时间网格预测。"""
    inputs = read_raw_signal_inputs(root, signal['columns'], timestamps, signal['imputation'])
    valid = np.isfinite(inputs).all(axis=1)
    prediction = np.full(len(timestamps), np.nan)
    if signal['backend'] == 'raw_tree':
        import lightgbm as lgb
        model = lgb.Booster(model_file=signal['weights'])
        prediction[valid] = model.predict(inputs[valid], num_threads=2)
    else:
        model = signal['model']
        prediction[valid] = inputs[valid] @ np.asarray(model['coefficients']) + model['intercept']
    return prediction


# 【核心逻辑232】分别按训练时点与整日时点产生信号，阈值只取固定训练样本。
def export_prediction_signals(root: Path, output: Path, date: str) -> list[dict]:
    """导出86,400个UTC秒，不读取当日y；训练取10,000个固定等距历史位置。"""
    folder = output / 'signals'
    folder.mkdir(parents=True, exist_ok=True)
    if not (output / 'reference_models.json').exists():
        fit_reference_signals(root, output)
    frozen_file = output / 'signal_pool_frozen.json'
    signals = json.loads(frozen_file.read_text()) if frozen_file.exists() else freeze_signal_pool(root, output)
    day = int(np.datetime64(date, 'ns').astype(np.int64))
    timestamps = day + np.arange(86400, dtype=np.int64) * SECOND
    train_start = int(np.datetime64('2025-05-10', 'ns').astype(np.int64))
    train_stop = int(np.datetime64('2026-06-01', 'ns').astype(np.int64)) - 301 * SECOND
    train_timestamps = np.linspace(train_start // SECOND, train_stop // SECOND, 10000, dtype=np.int64) * SECOND
    completed = []
    for signal in signals:
        metadata_file = folder / f'{signal["id"]}.json'
        if metadata_file.exists():
            completed.append(json.loads(metadata_file.read_text()))
            continue
        times = np.concatenate([train_timestamps, timestamps])
        if signal['backend'] in ['overnight', 'comparison']:
            prediction = predict_cached_model(root, signal, times)
        else:
            prediction = predict_raw_model(root, signal, times)
        training = prediction[:len(train_timestamps)]
        training = training[np.isfinite(training)]
        quantiles = np.quantile(np.abs(training), [.5, .8, .9])
        record = {'id': signal['id'], 'horizon_s': signal['horizon_s'], 'backend': signal['backend'],
                  'training_sample_count': len(training), 'train_abs_q50_bps': float(quantiles[0]),
                  'train_abs_q80_bps': float(quantiles[1]), 'train_abs_q90_bps': float(quantiles[2]),
                  'date': date, 'rows': len(timestamps), 'date_valid_rows': int(np.isfinite(prediction[len(train_timestamps):]).sum()),
                  'label_filter_applied': False, 'file': str(folder / f'{signal["id"]}.csv')}
        pl.DataFrame({'timestamp_ns': timestamps, 'prediction_bps': prediction[len(train_timestamps):]}).write_csv(record['file'])
        metadata_file.write_text(json.dumps(record, indent=2))
        completed.append(record)
        (output / 'signal_exports.json').write_text(json.dumps(completed, indent=2))
        print('SIGNAL_READY', signal['id'], record['date_valid_rows'], flush=True)
    return completed


# 【核心逻辑233】只用训练与验证期给OBI、OBI加净深度两个机制对照校准bps预测。
def fit_reference_signals(root: Path, output: Path) -> None:
    """固定10万训练秒和1万验证秒，未来标签仅用于校准模型，不决定当日信号行集。"""
    from mm_research.models.linear import linear_statistics, subset_statistics, fit_linear_model, score_linear_model
    train_start = int(np.datetime64('2025-05-10', 's').astype(np.int64))
    train_stop = int(np.datetime64('2026-06-01', 's').astype(np.int64)) - 301
    validation_start = int(np.datetime64('2026-06-01', 's').astype(np.int64))
    validation_stop = int(np.datetime64('2026-09-01', 's').astype(np.int64)) - 301
    training = np.linspace(train_start, train_stop, 100000, dtype=np.int64) * SECOND
    tuning = np.linspace(validation_start, validation_stop, 10000, dtype=np.int64) * SECOND
    timestamps = np.concatenate([training, tuning])
    values = read_raw_signal_inputs(root, [OBI, DEPTH], timestamps, {})
    labels = pl.scan_parquet(str(root / 'workspace/data/labels/server_mid_1s_v1/horizon=1s/*.parquet')).filter(
        pl.col('timestamp_ns').is_in(timestamps)
    ).select('timestamp_ns', 'label_valid', 'y_mid_1s_bps').collect(engine='streaming')
    np.testing.assert_array_equal(labels['timestamp_ns'].to_numpy(), timestamps)
    response = labels['y_mid_1s_bps'].to_numpy()
    valid = labels['label_valid'].to_numpy() & np.isfinite(values).all(axis=1)
    train = valid & (timestamps < validation_start * SECOND)
    validation = valid & (timestamps >= validation_start * SECOND)
    train_statistics = linear_statistics(values[train], response[train])
    validation_statistics = linear_statistics(values[validation], response[validation])
    models = {}
    for name, indices in [('obi_1s', [0]), ('obi_depth_1s', [0, 1])]:
        fit = subset_statistics(train_statistics, indices)
        tune = subset_statistics(validation_statistics, indices)
        candidates = [fit_linear_model(fit, 'ridge', alpha) for alpha in [.0001, .001, .01, .1, 1.]]
        model = min(candidates, key=lambda candidate: score_linear_model(tune, candidate)['mse_bps2'])
        model['columns'] = [[OBI, DEPTH][index] for index in indices]
        model['validation'] = score_linear_model(tune, model)
        models[name] = model
    (output / 'reference_models.json').write_text(json.dumps(models, indent=2))
