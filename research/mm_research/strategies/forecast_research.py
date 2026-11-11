"""四期限模型的训练期前向预测、幅度校准与冻结策略信号。"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import polars as pl
import torch
import yaml

from mm_research.data.storage import file_locks, write_json_atomic
from mm_research.models.input_comparison import materialize_new_column, select_new_inputs
from mm_research.models.target180 import Target180MLP, fit_cached_tensors
from mm_research.multihorizon.common import DAY, HORIZONS, SECOND, model_inputs


# 【非核心】加载明确的研究版本，固定全部模型来源和执行参数。
def research_context(filename: Path) -> tuple:
    """返回配置、原矩阵目录与独立结果目录，配置变化要求新版本。"""
    config = yaml.safe_load(filename.read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'config.json'
    with file_locks([frozen]):
        if frozen.exists():
            if json.loads(frozen.read_text()) != config:
                raise ValueError('策略研究配置变化，需要新的输出版本')
        else:
            write_json_atomic(frozen, config)
    return config, Path(config['source']), output


# 【核心逻辑320】检查完整事件分片和已有验收状态，按冻结日期建立共同回放范围。
def inventory_replay_days(filename: Path) -> None:
    """只检查配置中的日期，文件缺失明确中止，原始事件不重复校验摘要。"""
    config, source, output = research_context(filename)
    records = []
    for split, dates in config['dates'].items():
        for date in dates:
            manifest_file = source / 'l2/event_cache' / date / 'events/manifest.json'
            manifest = json.loads(manifest_file.read_text())
            if not manifest['full_day_validated']:
                raise ValueError(f'行情尚未通过完整日验收 {date}')
            for shard in manifest['shards']:
                file = manifest_file.parent / shard['file']
                if not file.is_file() or file.stat().st_size != shard['bytes']:
                    raise ValueError(f'已验收事件分片缺失或大小变化 {file}')
            records.append({'date': date, 'split': split, 'manifest': str(manifest_file),
                            'logical_events': manifest['logical_events']})
    write_json_atomic(output / 'replay_days.json', {'days': records})
    link = output / 'l2'
    if not link.exists():
        link.symlink_to((source / 'l2').resolve(), target_is_directory=True)


# 【非核心】旧输入沿用本折规范化缓存，新列在本折内部拟合前缀重新选择和规范化。
def oof_inputs(config: dict, source: Path, folder: Path, signal: dict, fold_id: str) -> tuple:
    """返回fit/tune缓存和完整前向区间输入，所有新窗口选择早于本折预测区间。"""
    fold = source / 'e1/folds' / fold_id
    fold_spec = next(item for item in json.loads((source / 'e0/splits.json').read_text())['oof_folds']
                     if item['id'] == fold_id)
    positions = {split: np.load(fold / f'{split}_rows.npy') for split in ['fit', 'tune']}
    positions['predict'] = np.arange(fold_spec['predict_start'], fold_spec['predict_stop'] + 300)
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    prep = np.load(fold / 'preprocessing.npz')
    raw = np.load(source / 'e0/x.npy', mmap_mode='r')
    inputs = {split: np.load(fold / f'{split}_x.npy', mmap_mode='r') for split in ['fit', 'tune']}
    cache_folder = Path('/tmp') / (Path(config['output']).name + '_oof') / fold_id
    cache_folder.mkdir(parents=True, exist_ok=True)
    predict_file = cache_folder / 'old_predict.npy'
    with file_locks([predict_file]):
        if not predict_file.exists():
            cache = np.lib.format.open_memmap(predict_file.with_suffix('.partial.npy'), mode='w+',
                                             dtype='float32', shape=(len(positions['predict']), raw.shape[1] * 2))
            for begin in range(0, len(cache), 65536):
                cache[begin:begin + 65536] = model_inputs(
                    raw, positions['predict'][begin:begin + 65536], prep['mean'], prep['scale'],
                )
                print('OOF_INPUT_ROWS', fold_id, min(begin + 65536, len(cache)), len(cache), flush=True)
            cache.flush()
            del cache
            predict_file.with_suffix('.partial.npy').replace(predict_file)
    inputs['predict'] = np.load(predict_file, mmap_mode='r')
    if signal['group'] == 'old':
        return inputs, positions, prep
    new_folder = Path(config['output']) / 'oof_inputs' / f'{signal["horizon_s"]}s' / fold_id
    new_folder.mkdir(parents=True, exist_ok=True)
    marker = new_folder / 'completed.json'
    with file_locks([marker]):
        if not marker.exists():
            selection = select_new_inputs(config, source, signal['horizon_s'], fold_id=fold_id)
            selection.write_csv(new_folder / 'selection.csv')
            caches = {split: np.lib.format.open_memmap(
                new_folder / f'{split}.partial.npy', mode='w+', dtype='float32',
                shape=(len(rows), selection.height * 2), fortran_order=True,
            ) for split, rows in positions.items()}
            tasks = [(index, row, positions, caches, manifest['start_timestamp_ns'], int(positions['predict'][-1]) + 1)
                     for index, row in enumerate(selection.iter_rows(named=True))]
            with ThreadPoolExecutor(max_workers=config['read_workers']) as readers:
                statistics = list(readers.map(materialize_new_column, tasks))
            for split, cache in caches.items():
                cache.flush()
                (new_folder / f'{split}.partial.npy').replace(new_folder / f'{split}.npy')
            write_json_atomic(marker, {'statistics': statistics})
    for split in inputs:
        inputs[split] = np.concatenate([inputs[split], np.load(new_folder / f'{split}.npy', mmap_mode='r')], axis=1)
    return inputs, positions, prep


# 【核心逻辑321】同一训练前缀独立拟合模型，完整下一段预测不使用其标签或有效性掩码。
def train_forward_signal(filename: Path, signal_id: str, fold_id: str) -> None:
    """MLP三种子取均值，树采用同参数CUDA XGBoost，检查点归属信号与折。"""
    config, source, output = research_context(filename)
    signal = next(item for item in config['signals'] if item['id'] == signal_id)
    folder = output / 'oof' / signal_id / fold_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'completed.json').exists():
        return
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    inputs, positions, prep = oof_inputs(config, source, folder, signal, fold_id)
    target_index = int(np.flatnonzero(HORIZONS == signal['horizon_s'])[0])
    target_mean = float(prep['target_mean'][target_index])
    target_scale = float(prep['target_scale'][target_index])
    targets = np.load(source / 'e0/y.npy', mmap_mode='r')
    labels = {split: ((targets[positions[split], target_index] - target_mean) / target_scale).astype('float32')
              for split in ['fit', 'tune']}
    predictors = []
    if signal['family'] == 'mlp':
        cached = {split: (torch.tensor(np.asarray(inputs[split]), device='cuda'),
                          torch.tensor(labels[split], device='cuda')) for split in ['fit', 'tune']}
        for seed in config['seeds']:
            torch.manual_seed(seed)
            trial = folder / f'seed{seed}'
            trial.mkdir(exist_ok=True)
            network = Target180MLP(inputs['fit'].shape[1]).cuda()
            predictors.append(fit_cached_tensors(network, cached, trial, config, config['learning_rate']))
        del cached
    else:
        import cupy as cp
        import xgboost as xgb
        from mm_research.models.tree180 import GpuBatchIterator, TreePredictor

        weights = folder / 'model.ubj'
        if weights.exists():
            booster = xgb.Booster(model_file=str(weights))
        else:
            train = xgb.QuantileDMatrix(GpuBatchIterator(inputs['fit'], labels['fit']), max_bin=64, nthread=2)
            tune = xgb.QuantileDMatrix(GpuBatchIterator(inputs['tune'], labels['tune']), max_bin=64, ref=train, nthread=2)
            parameters = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'tree_method': 'hist',
                          'device': 'cuda', 'max_depth': 6, 'max_bin': 64, 'min_child_weight': 500,
                          'eta': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'lambda': 10,
                          'seed': 17, 'nthread': 2}
            booster = xgb.train(parameters, train, num_boost_round=1000, evals=[(tune, 'tune')],
                                early_stopping_rounds=50, verbose_eval=25)
            booster = booster[:booster.best_iteration + 1]
            temporary = weights.with_suffix('.partial.ubj')
            booster.save_model(str(temporary))
            temporary.replace(weights)
            del train, tune
            cp.get_default_memory_pool().free_all_blocks()
        booster.set_param({'device': 'cuda', 'nthread': 2})
        predictors.append(TreePredictor(booster))
    torch.cuda.empty_cache()
    prediction = np.empty(len(positions['predict']), dtype='float64')
    for begin in range(0, len(prediction), 65536):
        stop = min(begin + 65536, len(prediction))
        values = torch.tensor(np.asarray(inputs['predict'][begin:stop]), device='cuda')
        with torch.no_grad():
            average = torch.stack([predictor(values) for predictor in predictors]).mean(dim=0)
        prediction[begin:stop] = average.cpu().numpy().astype('float64') * target_scale + target_mean
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    result = pl.DataFrame({'timestamp_ns': manifest['start_timestamp_ns'] + positions['predict'] * SECOND,
                           'prediction_log_bps': prediction,
                           'target_log_bps': np.asarray(targets[positions['predict'], target_index])})
    temporary = folder / 'predictions.partial.parquet'
    result.write_parquet(temporary, compression='zstd')
    temporary.replace(folder / 'predictions.parquet')
    write_json_atomic(folder / 'completed.json', {'status': 'success', 'signal': signal, 'fold': fold_id})


# 【非核心】按相同时间键逐批读取模型预测并取固定等权平均。
def ensemble_prediction(files: list[Path]) -> pl.DataFrame:
    """模型来源已在配置固定，行数和时间键必须一致，标签不参与组合权重。"""
    reference = pl.read_parquet(files[0])
    prediction = reference['prediction_log_bps'].to_numpy().copy()
    for file in files[1:]:
        current = pl.read_parquet(file, columns=['timestamp_ns', 'prediction_log_bps'])
        np.testing.assert_array_equal(reference['timestamp_ns'].to_numpy(), current['timestamp_ns'].to_numpy())
        prediction += current['prediction_log_bps'].to_numpy()
    return reference.with_columns(pl.Series('prediction_log_bps', prediction / len(files)))


# 【核心逻辑322】训练期前向预测估计校准与阈值，随后冻结两段验证的完整信号。
def calibrate_and_export(filename: Path) -> None:
    """校准为OLS幅度映射，分组、阈值和180/30秒同向过滤均由训练OOF定义。"""
    config, source, output = research_context(filename)
    export = output / 'signals'
    export.mkdir(exist_ok=True)
    calibrations = {}
    for signal in config['signals']:
        folder = output / 'oof' / signal['id']
        oof = pl.concat([pl.read_parquet(folder / fold / 'predictions.parquet')
                         for fold in config['folds']]).sort('timestamp_ns')
        # 每折末尾300秒在预测中保留，校准时排除，避免标签越过折预测边界。
        splits = json.loads((source / 'e0/splits.json').read_text())
        start = json.loads((source / 'e0/manifest.json').read_text())['start_timestamp_ns']
        usable = pl.lit(False)
        for fold in splits['oof_folds']:
            usable |= pl.col('timestamp_ns').is_between(start + fold['predict_start'] * SECOND,
                                                       start + fold['predict_stop'] * SECOND, closed='left')
        sample = oof.filter(usable & pl.col('target_log_bps').is_finite() & pl.col('prediction_log_bps').is_finite())
        stats = sample.select(pl.col('prediction_log_bps').mean().alias('x'),
                              pl.col('target_log_bps').mean().alias('y'),
                              pl.col('prediction_log_bps').var(ddof=0).alias('variance'),
                              pl.cov('prediction_log_bps', 'target_log_bps', ddof=0).alias('covariance')).row(0, named=True)
        slope = stats['covariance'] / stats['variance']
        intercept = stats['y'] - slope * stats['x']
        values = sample['prediction_log_bps'].to_numpy() * slope + intercept
        calibration = {'slope': slope, 'intercept': intercept, 'fit': 'train_forward_oof',
                       'train_mean_bps': stats['y'],
                       'entry_bps': float(np.quantile(np.abs(values), .8)),
                       'exit_bps': float(np.quantile(np.abs(values), .5)),
                       'scale_bps': float(np.quantile(np.abs(values), .9)),
                       'group_boundaries': np.quantile(values, np.arange(1, 10) / 10).tolist()}
        calibrations[signal['id']] = calibration
        for split in ['val_dev', 'val_select']:
            files = [Path(config['models']) / 'models' / model / f'{split}.parquet' for model in signal['models']]
            frame = ensemble_prediction(files).with_columns(
                (pl.col('prediction_log_bps') * slope + intercept).alias('prediction_bps'))
            frame.write_parquet(export / f'{signal["id"]}_{split}.parquet', compression='zstd')
            for date in config['dates'][split]:
                day = int(np.datetime64(date, 'ns').astype(np.int64))
                daily = frame.filter(pl.col('timestamp_ns').is_between(day, day + DAY * SECOND, closed='left'))
                if daily.height != DAY:
                    raise ValueError(f'完整交易日信号少行 {signal["id"]} {date}')
                daily.select('timestamp_ns', 'prediction_bps').write_csv(export / f'{signal["id"]}_{date}.csv')
    write_json_atomic(output / 'calibration.json', calibrations)
    diagnose_forecasts(config, source, output, calibrations)
    write_json_atomic(output / 'signals_completed.json', {'signals': len(config['signals']), 'status': 'success'})


# 【核心逻辑323】对照校准前后的误差和条件收益曲线，预测分组边界只来自训练OOF。
def diagnose_forecasts(config: dict, source: Path, output: Path, calibrations: dict) -> None:
    """输出六组信号两段验证的IC、R²、幅度与多期限分组收益，保留完整预测网格。"""
    export = output / 'signals'
    diagnostics = []
    scores = []
    labels = np.load(source / 'e0/y.npy', mmap_mode='r')
    start = json.loads((source / 'e0/manifest.json').read_text())['start_timestamp_ns']
    for signal in config['signals']:
        calibration = calibrations[signal['id']]
        for split in ['val_dev', 'val_select']:
            frame = pl.read_parquet(export / f'{signal["id"]}_{split}.parquet')
            valid = frame.filter(pl.col('evaluation_valid') & pl.col('target_log_bps').is_finite())
            for prediction_name in ['prediction_log_bps', 'prediction_bps']:
                prediction = pl.col(prediction_name)
                actual = pl.col('target_log_bps')
                record = valid.select(
                    pl.len().alias('rows'), pl.corr(prediction, actual).alias('ic'),
                    pl.corr(prediction, actual, method='spearman').alias('rank_ic'),
                    (prediction - actual).pow(2).mean().alias('mse_bps2'),
                    (1 - (prediction - actual).pow(2).sum() /
                     (actual - calibration['train_mean_bps']).pow(2).sum()).alias('r2_vs_train_oof_mean'),
                    prediction.std().alias('prediction_std_bps'),
                ).row(0, named=True)
                scores.append({'signal': signal['id'], 'split': split, 'prediction': prediction_name, **record})
            rows = ((frame['timestamp_ns'].to_numpy() - start) // SECOND).astype('int64')
            group = np.searchsorted(calibration['group_boundaries'], frame['prediction_bps'].to_numpy())
            table = frame.select('timestamp_ns', 'prediction_bps').with_columns(pl.Series('group', group))
            for horizon in [30, 60, 120, 180, 300]:
                index = int(np.flatnonzero(HORIZONS == horizon)[0])
                table = table.with_columns(pl.Series(f'y_{horizon}s', labels[rows, index]))
            grouped = table.group_by('group').agg(pl.len().alias('rows'), pl.col('prediction_bps').mean(),
                         *[pl.col(f'y_{horizon}s').filter(pl.col(f'y_{horizon}s').is_finite()).mean()
                           for horizon in [30, 60, 120, 180, 300]])
            diagnostics.append(grouped.with_columns(pl.lit(signal['id']).alias('signal'), pl.lit(split).alias('split')))
    pl.concat(diagnostics).write_parquet(output / 'signal_groups.parquet')
    pl.DataFrame(scores).write_csv(output / 'signal_calibration_scores.csv')
