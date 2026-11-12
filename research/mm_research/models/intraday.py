"""在现有高频输入上研究较低换手的30分钟与2小时预测，完整保留实际目标单位。"""

import fcntl
import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.models.cross_market import cache_group, publish_cache
from mm_research.multihorizon.common import context, SECOND


# 【核心逻辑361】直接从已有Mid序列构造长窗口状态和未来收益，目标边界按实际周期剔除。
def prepare_intraday(filename: Path) -> None:
    """输入现有每秒市场矩阵，输出18列过去状态、两个未来目标和原行集上的有效位置。"""
    _, config, output = context(filename)
    if (output / 'prepared.json').exists():
        return
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    segments = json.loads((source / 'e0/splits.json').read_text())['segments']
    stop = segments['val_select']['stop_row']
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    prices = np.full(stop, np.nan)
    prices[1:] = market[:stop - 1, 0]
    frame = pl.DataFrame({'price': prices}).with_columns(pl.col('price').fill_nan(None).log().alias('log_price'))
    frame = frame.with_columns(pl.col('log_price').diff().alias('return'))
    temporary = Path('/tmp/hft_intraday_features_20261111.npy')
    values = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32', shape=(stop, 18))
    columns = []
    for index, window in enumerate(config['state_windows_s']):
        minimum = int(window * .8)
        log_price = pl.col('log_price')
        volatility = log_price.rolling_std(window, min_samples=minimum)
        fields = [
            ((log_price - log_price.shift(window)) * 10000).alias(f'return_{window}s'),
            (pl.col('return').rolling_std(window, min_samples=minimum) * 10000).alias(f'volatility_{window}s'),
            pl.when(volatility > 0).then(
                (log_price - log_price.rolling_mean(window, min_samples=minimum)) / volatility,
            ).otherwise(None).alias(f'price_zscore_{window}s'),
        ]
        calculated = frame.select(fields)
        values[:, index * 3:index * 3 + 3] = calculated.to_numpy().astype('float32')
        columns.extend(calculated.columns)
    values.flush()
    del values, frame, prices
    publish_cache(temporary, output / 'cross.npy')
    write_json_atomic(output / 'features_ready.json', {'columns': columns, 'rows': stop})
    cache_group(filename, 'all')
    original_rows = {
        split: np.load(source / f'e1/folds/final/{split}_rows.npy') for split in ['fit', 'tune']
    }
    first_tune = int(original_rows['tune'][0])
    for horizon in config['horizons_s']:
        current = np.asarray(market[:stop, 0])
        future = np.asarray(market[horizon:stop + horizon, 0])
        valid = (np.isfinite(current) & np.isfinite(future) & (current > 0) & (future > 0)
                 & (market[:stop, 8] > .5) & (market[horizon:stop + horizon, 8] > .5)
                 & (market[:stop, 9] == market[horizon:stop + horizon, 9]))
        targets = np.full(stop, np.nan)
        targets[valid] = np.log(future[valid] / current[valid]) * 10000
        np.save(output / f'target_{horizon}s.npy', targets)
        positions = {}
        for split in ['fit', 'tune']:
            rows = original_rows[split]
            boundary = first_tune if split == 'fit' else segments['train']['stop_row']
            selected = np.flatnonzero(np.isfinite(targets[rows]) & (rows + horizon < boundary))
            np.save(output / f'{split}_positions_{horizon}s.npy', selected)
            positions[split] = selected
        fit_labels = targets[original_rows['fit'][positions['fit']]]
        write_json_atomic(output / f'target_{horizon}s.json', {
            'horizon_s': horizon, 'mean': float(fit_labels.mean()), 'scale': float(fit_labels.std()),
            'fit_rows': len(positions['fit']), 'tune_rows': len(positions['tune']),
            'purge_s': horizon, 'target': 'future_log_mid_return_bps',
        })
    write_json_atomic(output / 'prepared.json', {'status': 'success', 'feature_columns': columns})


# 【非核心】三份已归一化输入按块拼接，所有值及缺失通道沿用训练期定义。
def input_parts(config: dict, output: Path, split: str) -> list[np.ndarray]:
    """返回旧146列、现货16列、长状态18列的只读矩阵，均含对应缺失通道。"""
    return [
        np.load(Path(config['local_cache']) / f'{split}_x.npy', mmap_mode='r'),
        np.load(Path(config['spot_inputs']) / f'{split}_x.npy', mmap_mode='r'),
        np.load(output / 'all/inputs/180s' / f'{split}_x.npy', mmap_mode='r'),
    ]


# 【核心逻辑362】新目标使用原参数CUDA树，拟合和早停明确剔除跨越目标边界的行。
def train_intraday(filename: Path, horizon: int, seed: int) -> None:
    """实际目标为配置中的秒数，训练标签标准化仅使用对应拟合行。"""
    import cupy as cp
    import xgboost as xgb
    from mm_research.models.tree180 import GpuBatchIterator

    _, config, output = context(filename)
    folder = output / 'models' / f'h{horizon}_xgb_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'completed.json').exists():
        return
    source = Path(config['source'])
    metadata = json.loads((output / f'target_{horizon}s.json').read_text())
    targets = np.load(output / f'target_{horizon}s.npy', mmap_mode='r')
    weights = folder / 'model.ubj'
    with Path('/tmp/hft_mm_research_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if weights.exists():
            model = xgb.Booster(model_file=str(weights))
        else:
            matrices = {}
            temporary_files = []
            for split in ['fit', 'tune']:
                parts = input_parts(config, output, split)
                positions = np.load(output / f'{split}_positions_{horizon}s.npy')
                rows = np.load(source / f'e1/folds/final/{split}_rows.npy')[positions]
                temporary = Path('/tmp') / f'hft_intraday_{split}_{horizon}_{seed}.npy'
                values = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32',
                    shape=(len(positions), sum(part.shape[1] for part in parts)))
                for begin in range(0, len(positions), 65536):
                    selected = positions[begin:begin + 65536]
                    values[begin:begin + len(selected)] = np.concatenate([part[selected] for part in parts], axis=1)
                values.flush()
                labels = ((targets[rows] - metadata['mean']) / metadata['scale']).astype('float32')
                reference = matrices['fit'] if split == 'tune' else None
                matrices[split] = xgb.QuantileDMatrix(
                    GpuBatchIterator(values, labels), max_bin=64, ref=reference, nthread=2,
                )
                temporary_files.append(temporary)
                del values, parts
                cp.get_default_memory_pool().free_all_blocks()
            parameters = {
                'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'tree_method': 'hist',
                'device': 'cuda', 'max_depth': 6, 'max_bin': 64, 'min_child_weight': 500,
                'eta': .05, 'subsample': .8, 'colsample_bytree': .8, 'lambda': 10,
                'seed': seed, 'nthread': 2,
            }
            model = xgb.train(parameters, matrices['fit'], num_boost_round=1000,
                evals=[(matrices['tune'], 'tune')], early_stopping_rounds=50, verbose_eval=50)
            model = model[:model.best_iteration + 1]
            model.save_model(str(folder / 'model.partial.ubj'))
            (folder / 'model.partial.ubj').replace(weights)
            del matrices
            cp.get_default_memory_pool().free_all_blocks()
            for temporary in temporary_files:
                temporary.unlink()
        model.set_param({'device': 'cuda', 'nthread': 2})
        predict_intraday(config, output, folder, model, metadata)
    write_json_atomic(folder / 'completed.json', {'status': 'success', 'horizon_s': horizon, 'seed': seed})


# 【核心逻辑363】按完整时间网格输出长周期预测，目标只控制评价掩码。
def predict_intraday(config: dict, output: Path, folder: Path, model, metadata: dict) -> None:
    """发布两段完整预测以及Train内部预测分布，后者用于冻结交易门槛。"""
    import cupy as cp

    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    segments = json.loads((source / 'e0/splits.json').read_text())['segments']
    horizon = metadata['horizon_s']
    targets = np.load(output / f'target_{horizon}s.npy', mmap_mode='r')
    metrics = {}
    for split in ['tune', 'val_dev', 'val_select']:
        parts = input_parts(config, output, split)
        prediction = np.empty(len(parts[0]), dtype='float64')
        for begin in range(0, len(prediction), 65536):
            stop = min(begin + 65536, len(prediction))
            values = cp.asarray(np.concatenate([part[begin:stop] for part in parts], axis=1))
            prediction[begin:stop] = cp.asnumpy(model.inplace_predict(values)).astype('float64') * metadata['scale'] + metadata['mean']
        if split == 'tune':
            np.save(folder / 'train_tail_predictions.npy', prediction)
            continue
        bounds = segments[split]
        rows = np.arange(bounds['start_row'], bounds['stop_row'])
        actual = np.asarray(targets[rows])
        table = pl.DataFrame({
            'timestamp_ns': manifest['start_timestamp_ns'] + rows * SECOND,
            'prediction_log_bps': prediction, 'target_log_bps': actual,
            'evaluation_valid': np.isfinite(actual) & (rows + horizon < bounds['stop_row']),
        })
        table.write_parquet(folder / f'{split}.parquet', compression='zstd')
        metrics[split] = table.filter(pl.col('evaluation_valid')).select(
            pl.len().alias('rows'), pl.corr('prediction_log_bps', 'target_log_bps').alias('ic'),
            pl.corr('prediction_log_bps', 'target_log_bps', method='spearman').alias('rank_ic'),
            (pl.col('prediction_log_bps') - pl.col('target_log_bps')).pow(2).mean().alias('mse'),
        ).row(0, named=True)
        print('LONG_HORIZON_METRICS', horizon, split, metrics[split], flush=True)
    write_json_atomic(folder / 'metrics.json', metrics)


# 【核心逻辑364】发布三种子组合信号与训练期门槛，直接供冻结收益实验使用。
def export_intraday(filename: Path) -> None:
    """按真实目标命名h1800/h7200，保留每个种子的结果和全部预测时间点。"""
    from mm_research.strategies.forecast_research import ensemble_prediction

    _, config, output = context(filename)
    signals = output / 'signals'
    signals.mkdir(exist_ok=True)
    thresholds = {}
    metrics = []
    for horizon in config['horizons_s']:
        group = f'h{horizon}'
        folders = [output / 'models' / f'{group}_xgb_seed{seed}' for seed in [17, 42, 73]]
        tail = sum(np.load(folder / 'train_tail_predictions.npy') for folder in folders) / 3
        thresholds[group] = {str(quantile): float(np.quantile(np.abs(tail), quantile)) for quantile in config['quantiles']}
        for split in ['val_dev', 'val_select']:
            table = ensemble_prediction([folder / f'{split}.parquet' for folder in folders])
            table.write_parquet(signals / f'{group}_{split}.parquet', compression='zstd')
            result = table.filter(pl.col('evaluation_valid')).select(
                pl.corr('prediction_log_bps', 'target_log_bps').alias('ic'),
            ).row(0, named=True)
            metrics.append({'horizon_s': horizon, 'split': split, **result})
    profit = Path(config['profit_output'])
    profit.mkdir(parents=True, exist_ok=True)
    write_json_atomic(profit / 'thresholds.json', {'fit': 'train_internal_tune_prediction_distribution', 'groups': thresholds})
    summary = output / 'summary'
    summary.mkdir(exist_ok=True)
    pl.DataFrame(metrics).write_csv(summary / 'models.csv')
    write_json_atomic(output / 'completed.json', {'status': 'success', 'ensemble_metrics': metrics})
