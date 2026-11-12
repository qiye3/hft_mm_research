"""使用已冻结的模型和归一化参数补齐后续历史月份，逐批推理并复用已有验证预测。"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch

from mm_research.data.storage import file_locks, write_json_atomic
from mm_research.models.target180 import Target180MLP
from mm_research.multihorizon.common import HORIZONS, SECOND, model_inputs
from mm_research.strategies.forecast_research import research_context
from mm_research.strategies.tradeability import dataset_definition


# 【核心逻辑332】按既定64列和训练期统计量准备后续月份，新因子继续读取上一完整秒。
def prepare_historical_inputs(filename: Path) -> None:
    """只缓存后续区间的新输入，旧146列直接分批读取原矩阵，按列保存恢复点。"""
    config, source, output = research_context(filename)
    manifest, segments = dataset_definition(source)
    bounds = segments['test']
    origin = Path(config['models']) / 'inputs/180s'
    selection = pl.read_csv(origin / 'selection.csv')
    statistics = json.loads((origin / 'completed.json').read_text())['statistics']
    folder = output / 'inputs'
    folder.mkdir(exist_ok=True)
    destination = folder / 'historical_new.npy'
    marker = folder / 'completed_columns.json'
    if (folder / 'completed.json').exists():
        return
    completed = json.loads(marker.read_text())['columns'] if marker.exists() else []
    mode = 'r+' if destination.exists() else 'w+'
    cache = np.lib.format.open_memmap(
        destination, mode=mode, dtype='float32', fortran_order=True,
        shape=(bounds['stop_row'] - bounds['start_row'], selection.height * 2),
    )
    first = manifest['start_timestamp_ns'] + (bounds['start_row'] - 1) * SECOND
    stop = manifest['start_timestamp_ns'] + (bounds['stop_row'] - 1) * SECOND
    for index, record in enumerate(selection.iter_rows(named=True)):
        name = record['factor']
        if name in completed:
            continue
        normalization = statistics[index]
        if normalization['factor'] != name:
            raise ValueError('新因子顺序与训练参数不一致')
        source_file = Path(record['source_file'])
        with file_locks([source_file], shared=True):
            frame = pl.scan_parquet(source_file).filter(
                pl.col('timestamp_ns').is_between(first, stop, closed='left'),
            ).select(name).collect(engine='streaming')
        values = frame[name].to_numpy().astype('float32')
        if len(values) != len(cache):
            raise ValueError(f'后续历史因子缺少时间位置 {name}')
        missing = ~np.isfinite(values)
        normalized = (np.where(missing, normalization['mean'], values) - normalization['mean']) / normalization['scale']
        cache[:, index] = np.clip(normalized, -20, 20)
        cache[:, index + selection.height] = missing
        cache.flush()
        completed.append(name)
        write_json_atomic(marker, {'columns': completed})
        print('HISTORICAL_INPUT', len(completed), selection.height, name, flush=True)
    write_json_atomic(folder / 'completed.json', {
        'status': 'success', 'channels': cache.shape[1],
        'start_row': bounds['start_row'], 'stop_row': bounds['stop_row'],
        'normalization_source': str(origin / 'completed.json'),
    })


# 【非核心】加载实际训练权重，MLP种子固定等权，XGBoost沿用原生GPU预测器。
def frozen_predictors(config: dict, signal: dict, channels: int) -> list:
    """仅进入eval推理，权重、特征和组合系数均由已有实验清单指定。"""
    predictors = []
    for model_id in signal['models']:
        folder = Path(config['models']) / 'models' / model_id
        if signal['family'] == 'mlp':
            network = Target180MLP(channels).cuda()
            network.load_state_dict(torch.load(folder / 'model.pt', map_location='cuda', weights_only=True))
            network.eval()
            predictors.append(network)
        else:
            import xgboost as xgb
            from mm_research.models.tree180 import TreePredictor

            booster = xgb.Booster(model_file=str(folder / 'model.ubj'))
            booster.set_param({'device': 'cuda', 'nthread': 2})
            predictors.append(TreePredictor(booster))
    return predictors


# 【非核心】在已有验证预测上检查权重、特征顺序与反标准化，避免后续月份口径漂移。
def audit_frozen_prediction(config: dict, source: Path, signal: dict, predictors: list) -> dict:
    """前65536个原验证位置仅用于数值对照，误差上限为0.001bps。"""
    manifest, segments = dataset_definition(source)
    begin = segments['val_dev']['start_row']
    rows = np.arange(begin, begin + 65536)
    preprocessing = np.load(source / 'e1/folds/final/preprocessing.npz')
    raw = np.load(source / 'e0/x.npy', mmap_mode='r')
    values = model_inputs(raw, rows, preprocessing['mean'], preprocessing['scale'])
    if signal['group'] == 'combined':
        cached = np.load(Path(config['models']) / 'inputs/180s/val_dev_x.npy', mmap_mode='r')
        values = np.concatenate([values, cached[:len(rows)]], axis=1)
    with torch.inference_mode():
        tensors = torch.tensor(values, device='cuda')
        current = torch.stack([predictor(tensors) for predictor in predictors]).mean(dim=0).cpu().numpy()
    target_index = int(np.flatnonzero(HORIZONS == 180)[0])
    current = current.astype('float64') * preprocessing['target_scale'][target_index] + preprocessing['target_mean'][target_index]
    reference = pl.scan_parquet(
        Path(config['forecast_source']) / 'signals' / f'{signal["id"]}_val_dev.parquet',
    ).select('prediction_log_bps').head(len(rows)).collect()['prediction_log_bps'].to_numpy()
    difference = float(np.max(np.abs(current - reference)))
    np.testing.assert_allclose(current, reference, rtol=0, atol=.001)
    return {'rows': len(rows), 'maximum_absolute_difference_bps': difference, 'tolerance_bps': .001}


# 【核心逻辑333】冻结模型逐批预测7—10月完整网格，未来标签仅用于评价有效性。
def export_historical_signal(filename: Path, signal_id: str) -> None:
    """记录原始与校准后bps，额外预测窗口的输入、权重及校准参数均来自原训练期。"""
    config, source, output = research_context(filename)
    manifest, segments = dataset_definition(source)
    bounds = segments['test']
    signal = next(item for item in config['signals'] if item['id'] == signal_id)
    destination = output / 'signals' / f'{signal_id}_test.parquet'
    if destination.exists() and (output / 'signals' / f'{signal_id}_parity.json').exists():
        return
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    preprocessing = np.load(source / 'e1/folds/final/preprocessing.npz')
    target_index = int(np.flatnonzero(HORIZONS == 180)[0])
    raw = np.load(source / 'e0/x.npy', mmap_mode='r')
    labels = np.load(source / 'e0/y.npy', mmap_mode='r')
    new_inputs = None
    channels = raw.shape[1] * 2
    if signal['group'] == 'combined':
        new_inputs = np.load(output / 'inputs/historical_new.npy', mmap_mode='r')
        channels += new_inputs.shape[1]
    predictors = frozen_predictors(config, signal, channels)
    audit = audit_frozen_prediction(config, source, signal, predictors)
    write_json_atomic(output / 'signals' / f'{signal_id}_parity.json', audit)
    if destination.exists():
        return
    calibration = json.loads((output / 'thresholds.json').read_text())['signals'][signal_id]['calibration']
    temporary = destination.with_suffix('.partial.parquet')
    writer = None
    try:
        for begin in range(bounds['start_row'], bounds['stop_row'], 65536):
            stop = min(begin + 65536, bounds['stop_row'])
            rows = np.arange(begin, stop)
            values = model_inputs(raw, rows, preprocessing['mean'], preprocessing['scale'])
            if new_inputs is not None:
                values = np.concatenate([values, new_inputs[begin - bounds['start_row']:stop - bounds['start_row']]], axis=1)
            with torch.inference_mode():
                tensors = torch.tensor(values, device='cuda')
                prediction = torch.stack([predictor(tensors) for predictor in predictors]).mean(dim=0).cpu().numpy()
            raw_prediction = prediction.astype('float64') * preprocessing['target_scale'][target_index]
            raw_prediction += preprocessing['target_mean'][target_index]
            table = pl.DataFrame({
                'timestamp_ns': manifest['start_timestamp_ns'] + rows * SECOND,
                'prediction_log_bps': raw_prediction,
                'prediction_bps': raw_prediction * calibration['slope'] + calibration['intercept'],
                'evaluation_valid': np.isfinite(labels[rows, target_index]) & (rows < bounds['stop_row'] - 300),
            }).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
            writer.write_table(table)
            if (begin - bounds['start_row']) % (65536 * 16) == 0:
                print('HISTORICAL_PREDICTION', signal_id, begin, bounds['stop_row'], flush=True)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(destination)
    write_json_atomic(output / 'signals' / f'{signal_id}_test.completed.json', {
        'status': 'success', 'rows': bounds['stop_row'] - bounds['start_row'],
        'weights': signal['models'], 'training_cutoff': '2026-03-22',
        'historical_test_status': 'previously_inspected_historical_reassessment',
    })
