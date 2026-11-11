"""多模型共享的整秒研究矩阵，训练期拟合标准化与逐周期候选排序。"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import tempfile

import numpy as np
import polars as pl

from mm_research.data.storage import file_locks, read_metadata, scan_series
from mm_research.factors.dataset import factor_module
from mm_research.models.dataset import RAW_COLUMNS
from mm_research.evaluation.dataset import load_evaluation_targets

SECOND = 1_000_000_000
HORIZONS = [1, 10, 60, 180, 300]


# 【核心逻辑136】按组均匀挑选候选列，覆盖不同公式和窗口，选择过程不查看收益。
def experiment_columns(root: Path, config: dict) -> tuple[dict[str, str], list[dict]]:
    """返回列名到输入目录的映射，每组候选数量由实验配置明确限定。"""
    factor_parameters = {
        "min_fraction": .8,
        "tick_size": .1,
        "aggregation_windows_s": [10, 60, 180, 600],
        "mean_diff_windows_s": [[10, 60], [60, 180], [180, 600]],
    }
    columns = {name: config['feature_folder'] for name in RAW_COLUMNS}
    skipped = []
    for source in config['factor_sources']:
        for batch in source['batches']:
            names = list(factor_module(batch).factor_expressions(factor_parameters))
            missing = [name for name in names if not (root / source['folder'] / f'{name}.parquet').exists()]
            if missing:
                if config['skip_unavailable_groups']:
                    skipped.append({'batch': batch, 'folder': source['folder'], 'missing_factors': missing})
                    continue
                raise ValueError(f'因子组尚未完整发布 {batch}')
            indices = np.linspace(0, len(names) - 1, config['candidates_per_group'], dtype=int)
            for index in indices:
                columns[names[index]] = source['folder']
    return columns, skipped


# 【非核心】预读一天的正式输入与五周期标签，计算和跨日历史仍由主线程顺序处理。
def load_experiment_day(task: tuple) -> tuple:
    """只读共享锁保护的文件，返回固定顺序的数据和标签，不提前计算未来特征。"""
    root, config, columns, date, dataset = task
    groups = {}
    for name, source in columns.items():
        groups.setdefault(source, []).append(name)
    daily = None
    files = [root / source / f'{name}.parquet' for name, source in columns.items()]
    with file_locks(files, shared=True):
        for source, selected in groups.items():
            scan, _ = scan_series(root / source, selected, [date])
            scan = scan.filter(pl.col('timestamp_ns') % SECOND == 0).select('timestamp_ns', *selected)
            daily = scan if daily is None else daily.join(scan, on='timestamp_ns', how='left', maintain_order='left')
        table = daily.collect(engine='streaming')
    labels = [load_evaluation_targets(root / config['label_folder'], date, horizon * SECOND, dataset) for horizon in HORIZONS]
    return table, labels


# 【核心逻辑135】逐日关联候选与五周期标签，发布一次可复用的有限内存训练缓存。
def prepare_experiment_data(root: Path, config: dict, inputs: dict | None = None, derivations: dict | None = None) -> dict:
    """矩阵保留全部整秒位置，输入缺失保留NaN，标签先实施300秒分界隔离。"""
    folder = root / config['dataset_output']
    if inputs is None:
        columns, skipped = experiment_columns(root, config)
    else:
        columns, skipped = inputs, []
    definitions = {name: read_metadata(root / path / f'{name}.parquet') for name, path in columns.items()}
    dataset = next(iter(definitions.values()))['dataset']
    dates = dataset['dates']
    if any(definition['dataset'] != dataset for definition in definitions.values()):
        raise ValueError('模型候选来自不同的上游定义')
    definitions = {name: definition['definition'] for name, definition in definitions.items()}
    identity = {'columns': columns, 'definitions': definitions, 'dataset': dataset, 'train_end': config['train_end'],
                'validation_end': config['validation_end'], 'label_folder': config['label_folder'], 'purge_seconds': 300}
    if derivations is not None:
        identity['derivations'] = derivations
    if (folder / 'metadata.json').exists():
        metadata = json.loads((folder / 'metadata.json').read_text())
        if metadata['identity'] != identity:
            raise ValueError('模型矩阵定义变化，需要新的dataset_output')
        return metadata
    start = int(dt.datetime.fromisoformat(dates[0]).replace(tzinfo=dt.timezone.utc).timestamp())
    train_boundary = int(dt.datetime.combine(dt.date.fromisoformat(config['train_end']) + dt.timedelta(days=1), dt.time(), tzinfo=dt.timezone.utc).timestamp())
    validation_boundary = int(dt.datetime.combine(dt.date.fromisoformat(config['validation_end']) + dt.timedelta(days=1), dt.time(), tzinfo=dt.timezone.utc).timestamp())
    rows = len(dates) * 86400
    names = list(columns) + list(derivations or {})
    counts = np.zeros(len(names))
    sums = np.zeros(len(names))
    squares = np.zeros(len(names))
    paired_count = np.zeros((5, len(names)))
    paired_x = np.zeros_like(paired_count)
    paired_y = np.zeros_like(paired_count)
    paired_xx = np.zeros_like(paired_count)
    paired_yy = np.zeros_like(paired_count)
    paired_xy = np.zeros_like(paired_count)
    target_count = np.zeros(5)
    target_sum = np.zeros(5)
    target_square = np.zeros(5)
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.model-matrix-', dir=folder.parent) as temporary:
        staging = Path(temporary)
        read_workers = config['read_workers'] if inputs is not None else 1
        with (
            (staging / 'raw_x.npy').open('wb') as raw_stream,
            (staging / 'y.npy').open('wb') as target_stream,
            ThreadPoolExecutor(max_workers=read_workers) as readers,
        ):
            for stream, shape in [(raw_stream, (rows, len(names))), (target_stream, (rows, 5))]:
                np.lib.format.write_array_header_2_0(stream, {'descr': np.dtype('float32').str,
                                                             'fortran_order': False, 'shape': shape})
            history = None
            pending = deque(readers.submit(load_experiment_day, (root, config, columns, date, dataset))
                            for date in dates[:read_workers])
            for day, date in enumerate(dates):
                table, daily_labels = pending.popleft().result()
                if day + read_workers < len(dates):
                    pending.append(readers.submit(load_experiment_day,
                                   (root, config, columns, dates[day + read_workers], dataset)))
                timestamps = table['timestamp_ns'].to_numpy() // SECOND
                expected = start + day * 86400 + np.arange(86400)
                np.testing.assert_array_equal(timestamps, expected)
                if derivations is not None:
                    from mm_research.models.comparison_contract import comparison_derivations
                    current = table if history is None else history.vstack(table)
                    # 保留前日120秒原始输入，派生量跨日连续且不重复生成基础数据。
                    history = current.tail(120)
                    table = current.with_columns(comparison_derivations(derivations)).tail(86400)
                values = table.select(names).to_numpy()
                labels = []
                for label in daily_labels:
                    np.testing.assert_array_equal(label['timestamp_ns'].to_numpy() // SECOND, expected)
                    valid = label['label_valid'].to_numpy().copy()
                    end = label['label_end_timestamp_ns'].to_numpy() / SECOND
                    if date <= config['train_end']:
                        valid &= (timestamps < train_boundary - 300) & (end < train_boundary)
                    elif date <= config['validation_end']:
                        valid &= (timestamps < validation_boundary - 300) & (end < validation_boundary)
                    labels.append(np.where(valid, label['target'].to_numpy(), np.nan))
                target = np.column_stack(labels)
                raw_stream.write(np.asarray(values, dtype=np.float32, order='C').tobytes())
                target_stream.write(np.asarray(target, dtype=np.float32, order='C').tobytes())
                if date <= config['train_end']:
                    training = timestamps < train_boundary - 300
                    if derivations is not None:
                        training &= np.isfinite(target[:, 0])
                    observed = values[training]
                    finite = np.isfinite(observed)
                    counts += finite.sum(axis=0)
                    clean = np.where(finite, observed, 0)
                    sums += clean.sum(axis=0)
                    squares += (clean * clean).sum(axis=0)
                    for index in range(5):
                        response = target[training, index]
                        valid = finite & np.isfinite(response[:, None])
                        left = np.where(valid, observed, 0)
                        right = np.where(valid, response[:, None], 0)
                        paired_count[index] += valid.sum(axis=0)
                        paired_x[index] += left.sum(axis=0)
                        paired_y[index] += right.sum(axis=0)
                        paired_xx[index] += (left * left).sum(axis=0)
                        paired_yy[index] += (right * right).sum(axis=0)
                        paired_xy[index] += (left * right).sum(axis=0)
                        usable = response[np.isfinite(response)]
                        target_count[index] += len(usable)
                        target_sum[index] += usable.sum()
                        target_square[index] += usable @ usable
                print(f'模型矩阵完成 {date}', flush=True)
        if (counts[:6] == 0).any():
            raise ValueError('原始基线列在训练期完全缺失')
        retained = counts > 0
        excluded = [name for name, usable in zip(names, retained) if not usable]
        means = sums[retained] / counts[retained]
        scales = np.sqrt(np.maximum(squares[retained] / counts[retained] - means * means, 0))
        scales[scales < 1e-12] = 1
        matrix = np.load(staging / 'raw_x.npy', mmap_mode='r')
        # 共享盘按连续大块写入，避免可写mmap触发大量4KiB远程回写。
        with (staging / 'x.npy').open('wb') as normalized:
            np.lib.format.write_array_header_2_0(normalized, {'descr': np.dtype('float32').str,
                                                               'fortran_order': False, 'shape': (rows, int(retained.sum()))})
            for offset in range(0, rows, 250000):
                values = matrix[offset:offset + 250000, retained]
                transformed = np.clip((values - means) / scales, -20, 20).astype(np.float32)
                normalized.write(transformed.tobytes())
                if offset % 2500000 == 0:
                    print(f'模型矩阵标准化 {offset}/{rows}', flush=True)
        del matrix
        (staging / 'raw_x.npy').unlink()
        paired_count = paired_count[:, retained]
        paired_x = paired_x[:, retained]
        paired_y = paired_y[:, retained]
        paired_xx = paired_xx[:, retained]
        paired_yy = paired_yy[:, retained]
        paired_xy = paired_xy[:, retained]
        divisor = np.maximum(paired_count, 1)
        centered_xy = paired_xy - paired_x * paired_y / divisor
        denominator = np.sqrt(np.maximum(paired_xx - paired_x ** 2 / divisor, 0) * np.maximum(paired_yy - paired_y ** 2 / divisor, 0))
        correlations = np.divide(centered_xy, denominator, out=np.zeros_like(centered_xy), where=denominator > 0)
        target_scales = np.sqrt(np.maximum(target_square / target_count - (target_sum / target_count) ** 2, 0))
        names = [name for name, usable in zip(names, retained) if usable]
        metadata = {'identity': identity, 'skipped_factor_groups': skipped, 'excluded_all_missing_columns': excluded,
                    'rows': rows, 'columns': names, 'start_timestamp_seconds': start,
                    'train_stop': train_boundary - 300 - start, 'validation_start': train_boundary - start,
                    'validation_stop': validation_boundary - 300 - start, 'test_start': validation_boundary - start,
                    'mean_x': means.tolist(), 'scale_x': scales.tolist(), 'train_ic': [[float(value) if count > 0 else None for value, count in zip(values, counts)]
                                 for values, counts in zip(correlations, paired_count)],
                    'target_scale_bps': target_scales.tolist(), 'matrix_dtype': 'float32',
                    'candidate_selection': 'absolute_training_pooled_pearson_ic_after_group_coverage',
                    'input_missing_policy': 'training_mean_plus_missing_indicator', 'standardized_clip': 20}
        (staging / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        staging.replace(folder)
    return metadata


# 【核心逻辑139】从完整时间网格读取过去序列，缺失用训练均值及显式标记处理。
def history_batch(matrix, indices: np.ndarray, columns: np.ndarray, sequence: int) -> np.ndarray:
    """标准化矩阵的零对应训练均值，附加等宽缺失通道，不跨越未来时点。"""
    positions = indices[:, None] + np.arange(1 - sequence, 1)
    values = np.asarray(matrix[positions[:, :, None], columns[None, None, :]])
    missing = ~np.isfinite(values)
    values = np.nan_to_num(values, nan=0, copy=False)
    return np.concatenate([values, missing.astype(np.float32)], axis=-1)
