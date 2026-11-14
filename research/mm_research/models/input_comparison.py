"""新旧因子的四期限输入对照，复用既有标签、切分、GPU训练和原子检查点。"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
import yaml

from mm_research.data.storage import file_locks, write_json_atomic
from mm_research.models.target180 import Target180MLP, fit_cached_tensors
from mm_research.multihorizon.common import DAY, HORIZONS, SECOND, model_inputs


# 【非核心】冻结本轮配置，输出位置和临时缓存均由配置显式指定。
def experiment_context(filename: Path) -> tuple:
    """返回配置、原146列研究目录和本轮结果目录，参数变化时要求新版本。"""
    config = yaml.safe_load(filename.read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    contract = output / 'config.json'
    if contract.exists() and json.loads(contract.read_text()) != config:
        raise ValueError('输入对照配置变化，需要新的结果版本')
    write_json_atomic(contract, config)
    return config, Path(config['source']), output


# 【核心逻辑313】将旧输入的同一拟合行、早停行及整秒验证向量缓存到实例本地盘。
def prepare_old_inputs(filename: Path) -> None:
    """旧146列保持原训练前缀标准化与缺失通道，验证缓存只读取两段验证区间。"""
    config, source, output = experiment_context(filename)
    destination = Path(config['local_cache'])
    destination.mkdir(parents=True, exist_ok=True)
    marker = output / 'old_inputs.json'
    if marker.exists():
        return
    fold = source / 'e1/folds/final'
    for split in ['fit', 'tune']:
        temporary = destination / f'{split}_x.partial.npy'
        shutil.copyfile(fold / f'{split}_x.npy', temporary)
        temporary.replace(destination / f'{split}_x.npy')
    preprocessing = np.load(fold / 'preprocessing.npz')
    splits = json.loads((source / 'e0/splits.json').read_text())['segments']
    matrix = np.load(source / 'e0/x.npy', mmap_mode='r')
    for split in ['val_dev', 'val_select']:
        bounds = splits[split]
        temporary = destination / f'{split}_x.partial.npy'
        cached = np.lib.format.open_memmap(
            temporary, mode='w+', dtype='float32',
            shape=(bounds['stop_row'] - bounds['start_row'], matrix.shape[1] * 2),
        )
        for begin in range(bounds['start_row'], bounds['stop_row'], 65536):
            stop = min(begin + 65536, bounds['stop_row'])
            rows = np.arange(begin, stop)
            cached[begin - bounds['start_row']:stop - bounds['start_row']] = model_inputs(
                matrix, rows, preprocessing['mean'], preprocessing['scale'],
            )
        cached.flush()
        del cached
        temporary.replace(destination / f'{split}_x.npy')
        print('OLD_VALIDATION_READY', split, flush=True)
    write_json_atomic(marker, {'status': 'success', 'channels': matrix.shape[1] * 2})


# 【核心逻辑315】只用内部拟合前缀的日级Rank IC选窗，每个公式保留一个窗口后选64列。
def select_new_inputs(config: dict, source: Path, horizon: int, *, fold_id: str = 'final') -> pl.DataFrame:
    """四期限分别选列，排除旧146列同名因子，早停尾部和外部验证不参与排序。"""
    fold = source / 'e1/folds' / fold_id
    fit = np.load(fold / 'fit_rows.npy', mmap_mode='r')
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    last_fit_day = int(fit[-1]) // DAY
    cutoff = str(np.datetime64(manifest['dates'][0]) + np.timedelta64(last_fit_day, 'D'))
    inventory = pl.read_csv(Path(config['factor_results']) / 'summary/factor_inventory.csv')
    files = [Path(config['factor_results']) / 'evaluation' / row['batch'] / row['factor'] / 'daily_ic.parquet'
             for row in inventory.iter_rows(named=True)]
    daily = pl.scan_parquet(files).filter(
        (pl.col('horizon_ns') == horizon * SECOND) & (pl.col('sampling') == 'common')
        & (pl.col('date') < cutoff) & pl.col('rank_ic').is_finite()
    ).select('factor', 'date', 'rank_ic').collect(engine='streaming')
    scores = daily.group_by('factor').agg(
        pl.col('rank_ic').mean().alias('fit_mean_rank_ic'),
        pl.col('date').n_unique().alias('fit_days'),
    ).filter(pl.col('fit_days') >= int(np.ceil(last_fit_day * 0.8))).with_columns(
        pl.col('fit_mean_rank_ic').abs().alias('score'),
        pl.col('factor').str.replace(r'_(?:\d+s_)?\d+s$', '').alias('formula'),
    ).filter(~pl.col('factor').is_in(manifest['feature_names']))
    selected = scores.sort(['score', 'factor'], descending=[True, False]).unique(
        subset='formula', keep='first', maintain_order=True,
    ).head(config['new_features'])
    if selected.height != config['new_features']:
        raise ValueError(f'{horizon}秒训练前缀可用新因子不足指定数量')
    return selected.join(inventory.select('factor', 'source_file'), on='factor', maintain_order='left').with_columns(
        pl.lit(cutoff).alias('selection_before_utc_date'),
    )


# 【核心逻辑316】读取一个已验收因子并按上一完整秒对齐，归一化参数仅来自拟合行。
def materialize_new_column(task: tuple) -> dict:
    """将该因子的值与缺失标记写入四份缓存的独立列，任务间共享输入文件读取锁。"""
    index, record, input_rows, caches, timestamp_start, stop_row = task
    name = record['factor']
    filename = Path(record['source_file'])
    values = np.empty(stop_row, dtype='float32')
    position = 0
    with file_locks([filename], shared=True):
        for batch in pq.ParquetFile(filename).iter_batches(columns=['timestamp_ns', name], batch_size=262144):
            take = min(len(batch), stop_row - position)
            if take <= 0:
                break
            timestamps = batch.column(0).slice(0, take).to_numpy(zero_copy_only=False)
            expected = timestamp_start + np.arange(position, position + take) * SECOND
            np.testing.assert_array_equal(timestamps, expected)
            values[position:position + take] = batch.column(1).slice(0, take).to_numpy(zero_copy_only=False)
            position += take
    if position != stop_row:
        raise ValueError(f'新因子时间范围不完整 {name}')
    fit_values = values[input_rows['fit'] - 1]
    fit_values = fit_values[np.isfinite(fit_values)]
    mean = float(np.mean(fit_values, dtype=np.float64))
    scale = float(max(np.std(fit_values, dtype=np.float64), 1e-6))
    for split, rows in input_rows.items():
        for begin in range(0, len(rows), 262144):
            stop = min(begin + 262144, len(rows))
            current = values[rows[begin:stop] - 1]
            missing = ~np.isfinite(current)
            normalized = np.clip((np.where(missing, mean, current) - mean) / scale, -20, 20)
            caches[split][begin:stop, index] = normalized
            caches[split][begin:stop, index + caches[split].shape[1] // 2] = missing
    print('NEW_INPUT_READY', name, flush=True)
    return {'factor': name, 'mean': mean, 'scale': scale}


# 【核心逻辑314】按期限生产可复用新输入缓存，保持与旧模型完全相同的训练和验证位置。
def prepare_new_inputs(filename: Path, horizon: int) -> None:
    """单列读取可并行，按列连续写缓存，发布完成标记后GPU任务才可读取。"""
    config, source, output = experiment_context(filename)
    destination = output / 'inputs' / f'{horizon}s'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'completed.json').exists():
        return
    selected = select_new_inputs(config, source, horizon)
    selected.write_csv(destination / 'selection.csv')
    fold = source / 'e1/folds/final'
    rows = {split: np.load(fold / f'{split}_rows.npy') for split in ['fit', 'tune']}
    bounds = json.loads((source / 'e0/splits.json').read_text())['segments']
    for split in ['val_dev', 'val_select']:
        rows[split] = np.arange(bounds[split]['start_row'], bounds[split]['stop_row'])
    stop_row = bounds['val_select']['stop_row']
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    caches = {
        split: np.lib.format.open_memmap(
            destination / f'{split}_x.partial.npy', mode='w+', dtype='float32',
            shape=(len(positions), selected.height * 2), fortran_order=True,
        ) for split, positions in rows.items()
    }
    tasks = [(index, record, rows, caches, manifest['start_timestamp_ns'], stop_row)
             for index, record in enumerate(selected.iter_rows(named=True))]
    with ThreadPoolExecutor(max_workers=config['read_workers']) as readers:
        statistics = list(readers.map(materialize_new_column, tasks))
    for split, cache in caches.items():
        cache.flush()
        (destination / f'{split}_x.partial.npy').replace(destination / f'{split}_x.npy')
    write_json_atomic(destination / 'completed.json', {
        'status': 'success', 'horizon_s': horizon, 'statistics': statistics,
        'selection': 'abs_mean_daily_rank_ic_fit_prefix_one_window_per_formula',
    })


# 【非核心】按实验组读取对应规范化缓存，组合组沿固定列序合并。
def load_input_group(config: dict, output: Path, horizon: int, group: str, split: str) -> np.ndarray:
    """旧列保留原次序，新列保留选列CSV次序，所有模型共用相同缺失标记。"""
    folders = {'old': Path(config['local_cache']), 'new': output / 'inputs' / f'{horizon}s'}
    selection = None
    if 'input_selection' in config:
        specification = config['input_selection']
        folders['new'] = Path(specification['source'])
        selection = json.loads(Path(specification['file']).read_text())['matrix_columns']
    parts = ['old', 'new'] if group == 'combined' else [group]
    matrices = []
    for part in parts:
        matrix = np.load(folders[part] / f'{split}_x.npy', mmap_mode='r')
        if part == 'new' and selection is not None:
            matrix = matrix[:, selection]
        matrices.append(matrix)
    if len(matrices) == 1:
        return matrices[0]
    return np.concatenate(matrices, axis=1)


# 【核心逻辑318】三种输入组按同一整秒位置输出预测，按UTC日及三个非重叠相位评价。
def predict_input_group(predictor, config: dict, source: Path, output: Path, folder: Path,
                        horizon: int, group: str, target_mean: float, target_scale: float) -> None:
    """预测完整验证网格，标签只决定评价行，逐批生成Parquet以控制内存。"""
    splits = json.loads((source / 'e0/splits.json').read_text())['segments']
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    labels = np.load(source / 'e0/y.npy', mmap_mode='r')
    target_index = int(np.flatnonzero(HORIZONS == horizon)[0])
    metrics = {}
    for split in ['val_dev']:
        bounds = splits[split]
        # 验证时分别映射旧/新缓存，组合仅在单批内分配。
        parts = ['old', 'new'] if group == 'combined' else [group]
        inputs = [load_input_group(config, output, horizon, part, split) for part in parts]
        temporary = folder / f'{split}.partial.parquet'
        writer = None
        try:
            for begin in range(0, len(inputs[0]), 65536):
                stop = min(begin + 65536, len(inputs[0]))
                values = np.concatenate([part[begin:stop] for part in inputs], axis=1)
                rows = np.arange(bounds['start_row'] + begin, bounds['start_row'] + stop)
                with torch.no_grad():
                    prediction = predictor(torch.tensor(values, device='cuda')).cpu().numpy()
                actual = np.asarray(labels[rows, target_index])
                table = pl.DataFrame({
                    'timestamp_ns': manifest['start_timestamp_ns'] + rows * SECOND,
                    'prediction_log_bps': prediction.astype('float64') * target_scale + target_mean,
                    'target_log_bps': actual,
                    'evaluation_valid': np.isfinite(actual) & (rows < bounds['stop_row'] - 300),
                }).to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        temporary.replace(folder / f'{split}.parquet')
        frame = pl.read_parquet(folder / f'{split}.parquet').filter(pl.col('evaluation_valid'))
        prediction = pl.col('prediction_log_bps')
        actual = pl.col('target_log_bps')
        statistics = [
            pl.len().alias('rows'), pl.corr(prediction, actual).alias('ic'),
            pl.corr(prediction, actual, method='spearman').alias('rank_ic'),
            (1 - (prediction - actual).pow(2).sum() / (actual - target_mean).pow(2).sum()).alias('r2'),
        ]
        phases = {}
        for phase in [0, horizon // 3, horizon * 2 // 3]:
            phases[str(phase)] = frame.filter(
                (pl.col('timestamp_ns') // SECOND) % horizon == phase,
            ).select(statistics).row(0, named=True)
        metrics[split] = {'all': frame.select(statistics).row(0, named=True), 'phases': phases}
        frame.with_columns((pl.col('timestamp_ns') // (DAY * SECOND)).alias('utc_day')).group_by(
            'utc_day',
        ).agg(statistics).sort('utc_day').write_parquet(folder / f'{split}_daily.parquet')
        print('VALIDATION', split, metrics[split], flush=True)
    write_json_atomic(folder / 'metrics.json', metrics)


# 【核心逻辑317】固定超参数对照三种输入，MLP复用逐epoch恢复，树模型采用CUDA分箱。
def train_input_group(filename: Path, horizon: int, group: str, family: str, seed: int) -> None:
    """模型只拟合原Train前缀，尾部14天早停，本轮仅评价开发期；冻结复核独立执行。"""
    config, source, output = experiment_context(filename)
    folder = output / 'models' / f'{horizon}s_{group}_{family}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'completed.json').exists():
        return
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    fold = source / 'e1/folds/final'
    prep = np.load(fold / 'preprocessing.npz')
    target_index = int(np.flatnonzero(HORIZONS == horizon)[0])
    target_mean = float(prep['target_mean'][target_index])
    target_scale = float(prep['target_scale'][target_index])
    labels = np.load(source / 'e0/y.npy', mmap_mode='r')
    cached = {}
    for split in ['fit', 'tune']:
        rows = np.load(fold / f'{split}_rows.npy', mmap_mode='r')
        values = load_input_group(config, output, horizon, group, split)
        normalized_labels = ((labels[rows, target_index] - target_mean) / target_scale).astype('float32')
        if family == 'mlp':
            cached[split] = (torch.tensor(np.asarray(values), device='cuda'),
                             torch.tensor(normalized_labels, device='cuda'))
        else:
            cached[split] = (values, normalized_labels)
        del values
    channels = cached['fit'][0].shape[1]
    if family == 'mlp':
        predictor = Target180MLP(channels).cuda()
        predictor = fit_cached_tensors(predictor, cached, folder, config, config['learning_rate'])
    else:
        import cupy as cp
        import xgboost as xgb
        from mm_research.models.tree180 import GpuBatchIterator, TreePredictor

        weights = folder / 'model.ubj'
        if weights.exists():
            booster = xgb.Booster(model_file=str(weights))
        else:
            matrices = {}
            for split in ['fit', 'tune']:
                iterator = GpuBatchIterator(*cached[split])
                reference = matrices['fit'] if split == 'tune' else None
                matrices[split] = xgb.QuantileDMatrix(iterator, max_bin=64, ref=reference, nthread=2)
                cp.get_default_memory_pool().free_all_blocks()
            parameters = {
                'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                'tree_method': 'hist', 'device': 'cuda', 'max_depth': 6, 'max_bin': 64,
                'min_child_weight': 500, 'eta': 0.05, 'subsample': 0.8,
                'colsample_bytree': 0.8, 'lambda': 10, 'seed': seed, 'nthread': 2,
            }
            booster = xgb.train(parameters, matrices['fit'], num_boost_round=1000,
                                evals=[(matrices['tune'], 'tune')], early_stopping_rounds=50, verbose_eval=25)
            booster = booster[:booster.best_iteration + 1]
            temporary = folder / 'model.partial.ubj'
            booster.save_model(str(temporary))
            temporary.replace(weights)
            del matrices
        booster.set_param({'device': 'cuda', 'nthread': 2})
        predictor = TreePredictor(booster)
    del cached
    torch.cuda.empty_cache()
    predict_input_group(predictor, config, source, output, folder, horizon, group, target_mean, target_scale)
    write_json_atomic(folder / 'completed.json', {
        'status': 'success', 'horizon_s': horizon, 'group': group, 'family': family,
        'seed': seed, 'channels': channels, 'elapsed_seconds': time.monotonic() - started,
    })


# 【非核心】按显式来源复用已经完成的180秒同参数基线，并记录来源路径。
def reuse_old180(filename: Path, family: str, seed: int, origin: Path) -> None:
    """核对数据源、目标和训练参数后，转换预测列名并复用原权重。"""
    config, source, output = experiment_context(filename)
    contract = json.loads((origin / 'contract.json').read_text())
    completed = json.loads((origin / 'completed.json').read_text())
    if completed['status'] != 'success' or contract['target'] != '180s_log_bps' or contract['seed'] != seed:
        raise ValueError('指定旧模型的目标、种子或完成状态不符合本轮基线')
    if Path(contract['config']['source']) != source:
        raise ValueError('指定旧模型的数据源与本轮不同')
    if family == 'mlp':
        parameters = ['batch_size', 'max_epochs', 'patience']
        if any(contract['config'][name] != config[name] for name in parameters):
            raise ValueError('指定旧MLP的训练参数与本轮不同')
        if contract['learning_rate'] != config['learning_rate']:
            raise ValueError('指定旧MLP的学习率与本轮不同')
    folder = output / 'models' / f'180s_old_{family}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'completed.json').exists():
        return
    metrics = json.loads((origin / 'metrics.json').read_text())
    for split in ['val_dev', 'val_select']:
        for statistics in [metrics[split]['all'], *metrics[split]['phases'].values()]:
            statistics['r2'] = statistics.pop('r2_vs_train_mean')
        temporary = folder / f'{split}.partial.parquet'
        pl.scan_parquet(origin / f'{split}.parquet').rename({
            'prediction_180s_log_bps': 'prediction_log_bps',
            'target_180s_log_bps': 'target_log_bps',
        }).sink_parquet(temporary, compression='zstd')
        temporary.replace(folder / f'{split}.parquet')
        pl.read_parquet(origin / f'{split}_daily.parquet').rename({
            'r2_vs_train_mean': 'r2',
        }).write_parquet(folder / f'{split}_daily.parquet')
    weight = 'model.pt' if family == 'mlp' else 'model.ubj'
    (folder / weight).symlink_to((origin / weight).resolve())
    write_json_atomic(folder / 'metrics.json', metrics)
    write_json_atomic(folder / 'completed.json', {
        'status': 'success', 'horizon_s': 180, 'group': 'old', 'family': family,
        'seed': seed, 'reused_from': str(origin),
    })


# 【核心逻辑319】汇总全部对照与逐日配对增量，失败清单进入最终报告。
def summarize_input_models(filename: Path) -> None:
    """按同周期、同模型和同种子比较组合相对旧输入的日均IC及Bootstrap区间。"""
    config, source, output = experiment_context(filename)
    records = []
    missing = []
    paired = []
    for horizon in config['horizons_s']:
        for group in ['old', 'new', 'combined']:
            for family, seeds in [('mlp', config['seeds']), ('xgb', [config['seeds'][0]])]:
                for seed in seeds:
                    name = f'{horizon}s_{group}_{family}_seed{seed}'
                    folder = output / 'models' / name
                    if not (folder / 'completed.json').exists():
                        missing.append(name)
                        continue
                    metrics = json.loads((folder / 'metrics.json').read_text())
                    for split, values in metrics.items():
                        records.append({'model': name, 'horizon_s': horizon, 'group': group,
                                        'family': family, 'seed': seed, 'split': split, **values['all']})
                        baseline = output / 'models' / f'{horizon}s_old_{family}_seed{seed}'
                        if group != 'combined' or not (baseline / 'completed.json').exists():
                            continue
                        daily = pl.read_parquet(folder / f'{split}_daily.parquet').join(
                            pl.read_parquet(baseline / f'{split}_daily.parquet'), on='utc_day', suffix='_old',
                        ).filter(pl.col('ic').is_finite() & pl.col('ic_old').is_finite())
                        differences = (daily['ic'] - daily['ic_old']).to_numpy()
                        generator = np.random.default_rng(180)
                        estimates = np.array([
                            differences[generator.integers(0, len(differences), len(differences))].mean()
                            for _ in range(2000)
                        ])
                        paired.append({'model': name, 'split': split, 'days': len(differences),
                                       'daily_ic_increment': float(differences.mean()),
                                       'ci95_low': float(np.quantile(estimates, 0.025)),
                                       'ci95_high': float(np.quantile(estimates, 0.975))})
    summary = output / 'summary'
    summary.mkdir(exist_ok=True)
    pl.DataFrame(records).write_csv(summary / 'models.csv')
    pl.DataFrame(paired).write_csv(summary / 'paired_daily_increment.csv')
    means = pl.DataFrame(records).group_by('horizon_s', 'group', 'family', 'split').agg(
        pl.col('ic').mean(), pl.col('rank_ic').mean(), pl.col('r2').mean(),
    ).sort(['horizon_s', 'group', 'family', 'split'])
    lines = ['# 新旧因子四期限模型对照', '',
             '主目标180秒，同时训练30、60、120秒对数收益bps。',
             '旧146列、新64列及新旧组合使用共同训练行和验证网格。',
             '新列按各期限内部拟合前缀日均Rank IC选窗和排序，内部尾部14天早停。',
             'Train截至2026-03-22，两段验证分别截至2026-05-14、2026-07-06。', '',
             '| 周期秒 | 输入 | 模型 | 分区 | 平均IC | 平均Rank IC | 平均R² |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in means.iter_rows(named=True):
        lines.append(f"| {row['horizon_s']} | {row['group']} | {row['family']} | {row['split']} "
                     f"| {row['ic']:.6f} | {row['rank_ic']:.6f} | {row['r2']:.6f} |")
    lines.extend(['', '逐种子指标见models.csv，组合相对旧输入的日级配对增量与95%区间见paired_daily_increment.csv。'])
    if missing:
        lines.extend(['', '未完成实验', *[f'- {name}' for name in missing]])
    (summary / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json_atomic(summary / 'completed.json', {'models': len(records) // 2, 'missing': missing})
    if missing:
        raise RuntimeError('模型对照存在未完成实验，详见报告')
