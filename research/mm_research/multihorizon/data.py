"""从正式研究列生成多期限对数标签、时间切分与无未来信息的共享矩阵。"""

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import datetime as dt
import json
from pathlib import Path
import re

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from mm_research.data.storage import read_metadata, scan_series, file_locks, write_json_atomic
from mm_research.factors.dataset import factor_module
from mm_research.models.dataset import RAW_COLUMNS
from mm_research.multihorizon.common import HORIZONS, SECOND, DAY, context, definition_digest

MARKET_COLUMNS = ['mid_price', 'bid_price_1', 'ask_price_1', 'bid_quantity_1', 'ask_quantity_1',
                  'book_exchange_ns', 'last_book_timestamp_ns', 'snapshot_age_ns', 'book_valid', 'book_segment_id']


# 【核心逻辑252】按构造组均匀选取固定候选，不读取任何旧验证、测试IC或模型选列。
def freeze_inputs(root: Path, config: dict) -> tuple[dict, list[dict], dict]:
    """返回源文件映射、逐列因果血缘与已有正式数据定义。"""
    factor_parameters = {
        "min_fraction": .8,
        "tick_size": .1,
        "aggregation_windows_s": [10, 60, 180, 600],
        "mean_diff_windows_s": [[10, 60], [60, 180], [180, 600]],
    }
    source_config = yaml.safe_load((root / config['source_config']).read_text())
    locations = {name: config['feature_folder'] for name in RAW_COLUMNS}
    expressions = {}
    for source in source_config['factor_sources']:
        for batch in source['batches']:
            formulas = factor_module(batch).factor_expressions(factor_parameters)
            names = list(formulas)
            chosen = np.linspace(0, len(names) - 1, config['factors_per_group'], dtype=int)
            for index in chosen:
                name = names[index]
                locations[name] = source['folder']
                expressions[name] = formulas[name]
    lineage = []
    book_fields = [f'{side}_{field}_{level}' for level in range(1, 11)
                   for side in ['bid', 'ask'] for field in ['price', 'quantity']]
    trade_fields = ['exchange_timestamp_ns', 'price', 'quantity', 'side', 'trade_id', 'quality_flags']
    for name, folder in locations.items():
        metadata = read_metadata(root / folder / f'{name}.parquet')
        if name in expressions:
            dependencies = expressions[name].meta.root_names()
            batch = metadata['definition']['batch']
        else:
            dependencies = [name]
            batch = 'features'
        # 表达式依赖保留精确根列；原始字段范围按相关Features定义展开并保留审计路径。
        roots = ' '.join(dependencies)
        uses_trade = any(token in roots for token in ['trade', 'volume', 'notional', 'flow', 'buy_', 'sell_'])
        uses_book = any(token in roots for token in ['book', 'bid', 'ask', 'imbalance_', 'mid', 'price', 'return', 'spread', 'micro'])
        raw_fields = []
        if uses_book:
            raw_fields.extend(book_fields)
        if uses_trade:
            raw_fields.extend(trade_fields)
        if not raw_fields:
            raw_fields = book_fields + trade_fields
        windows = [int(value) for value in re.findall(r'(\d+)s', name)]
        lineage.append({
            'feature_name': name, 'meaning': name, 'source_file': f'{folder}/{name}.parquet',
            'expression_dependencies': ','.join(dependencies), 'raw_fields_used': ','.join(raw_fields),
            'raw_field_resolution': 'conservative_source_scope_with_exact_expression_dependencies',
            'lookback_ms': max(windows, default=0) * 1000,
            'max_source_ts_rule': 'source_exchange_ts + 20ms <= feature_grid_ts',
            'ready_ts_rule': 'feature_grid_ts + 1ms; decision uses previous completed grid',
            'code_location': f'research/mm_research/factors/{batch}.py' if batch != 'features' else 'research/mm_research/features/',
        })
    reference = read_metadata(root / config['feature_folder'] / 'mid_price.parquet')['dataset']
    return locations, lineage, reference


# 【非核心】一天输入只读取同一正式版本，保持午夜前最后一秒特征供下一日可用时钟使用。
def read_day(task: tuple) -> pl.DataFrame:
    """返回整秒市场状态和原始候选，源Parquet共享锁持续到物化完成。"""
    root, date, locations, feature_folder = task
    groups = {}
    for name, folder in locations.items():
        groups.setdefault(folder, []).append(name)
    groups[feature_folder] = list(dict.fromkeys([*MARKET_COLUMNS, *groups[feature_folder]]))
    filenames = [root / folder / f'{name}.parquet' for folder, names in groups.items() for name in names]
    result = None
    with file_locks(filenames, shared=True):
        for folder, names in groups.items():
            query, _ = scan_series(root / folder, names, [date])
            query = query.filter(pl.col('timestamp_ns') % SECOND == 0).select('timestamp_ns', *names)
            table = query.collect(engine='streaming')
            result = table if result is None else result.join(table, on='timestamp_ns', how='left', maintain_order='left')
    return result


# 【核心逻辑253】按完整UTC日期冻结60/20/20及前向OOF，全部分界剔除300秒未来标签。
def make_splits(dates: list[str]) -> dict:
    """返回四个互斥区间及三个扩展训练折，Test标记为已接触历史回看。"""
    count = len(dates)
    train_days = int(count * .6)
    validation_days = int(count * .2)
    dev_days = validation_days // 2
    boundaries = [0, train_days, train_days + dev_days, train_days + validation_days, count]
    segments = {}
    start = int(np.datetime64(dates[0], 's').astype(np.int64))
    for index, name in enumerate(['train', 'val_dev', 'val_select', 'test']):
        begin, end = boundaries[index:index + 2]
        segments[name] = {'start_row': begin * DAY, 'stop_row': end * DAY,
                          'start_utc': str(np.datetime64(start + begin * DAY, 's')),
                          'end_utc_exclusive': str(np.datetime64(start + end * DAY, 's')),
                          'purge_seconds': 300, 'dropped_tail_rows': 300}
    points = [int(train_days * fraction) * DAY for fraction in [.4, .6, .8, 1.]]
    folds = [{'id': f'oof_{index}', 'train_stop': points[index] - 300,
              'predict_start': points[index], 'predict_stop': points[index + 1] - 300}
             for index in range(3)]
    return {'segments': segments, 'oof_folds': folds, 'test_status': 'previously_inspected_historical_reassessment'}


# 【核心逻辑254】累计标签只跨越完整连续有效盘口，分段标签由相邻累计差值精确形成。
def cumulative_labels(mid: np.ndarray, valid: np.ndarray, segment: np.ndarray,
                      horizons: np.ndarray = HORIZONS) -> np.ndarray:
    """返回Float64对数收益bps；缺口、来源边界和序列末尾保留NaN。"""
    result = np.full((len(mid), len(horizons)), np.nan)
    bad_prefix = np.r_[0, np.cumsum(~valid)]
    for index, horizon in enumerate(horizons):
        if horizon >= len(mid):
            continue
        current = np.arange(len(mid) - horizon)
        future = current + horizon
        usable = (bad_prefix[future + 1] - bad_prefix[current] == 0) & (segment[current] == segment[future])
        rows = current[usable]
        result[rows, index] = np.log(mid[rows + horizon] / mid[rows]) * 10000
    return result


# 【核心逻辑255】逐日发布市场状态和因果特征缓存，跨日标签保留300秒未来市场检查区。
def prepare_dataset(filename: Path) -> None:
    """输入正式单列Parquet，输出E0研究表、只读训练矩阵和冻结QA，不读取旧标签。"""
    root, config, output = context(filename)
    folder = output / 'e0'
    folder.mkdir(exist_ok=True)
    if (folder / 'manifest.json').exists():
        return
    locations, lineage, reference = freeze_inputs(root, config)
    dates = reference['dates']
    splits = make_splits(dates)
    write_json_atomic(folder / 'splits.json', splits)
    split_hash = definition_digest(splits)
    pl.DataFrame(lineage).write_csv(folder / 'feature_lineage.csv')
    names = list(locations)
    count = len(dates) * DAY
    matrix = np.lib.format.open_memmap(folder / 'x.npy', mode='w+', dtype='float32', shape=(count, len(names)))
    market = np.lib.format.open_memmap(folder / 'market.npy', mode='w+', dtype='float64', shape=(count, len(MARKET_COLUMNS)))
    timestamps = np.lib.format.open_memmap(folder / 'timestamp_ns.npy', mode='w+', dtype='int64', shape=(count,))
    previous_features = np.full(len(names), np.nan)
    tasks = [(root, date, locations, config['feature_folder']) for date in dates]
    with ThreadPoolExecutor(max_workers=2) as readers:
        pending = deque(readers.submit(read_day, task) for task in tasks[:2])
        for day, date in enumerate(dates):
            table = pending.popleft().result()
            if day + 2 < len(tasks):
                pending.append(readers.submit(read_day, tasks[day + 2]))
            expected = int(np.datetime64(date, 'ns').astype(np.int64)) + np.arange(DAY) * SECOND
            np.testing.assert_array_equal(table['timestamp_ns'].to_numpy(), expected)
            raw_features = table.select(names).to_numpy()
            shifted = np.vstack([previous_features, raw_features[:-1]])
            previous_features = raw_features[-1].copy()
            row = slice(day * DAY, (day + 1) * DAY)
            matrix[row] = shifted
            market[row] = table.select(MARKET_COLUMNS).to_numpy()
            timestamps[row] = expected
            if day % 10 == 0:
                write_json_atomic(folder / 'progress.json', {'days': day + 1, 'total_days': len(dates)})
                print('E0_DAY', day + 1, date, flush=True)
    matrix.flush()
    market.flush()
    timestamps.flush()
    targets = np.lib.format.open_memmap(folder / 'y.npy', mode='w+', dtype='float64', shape=(count, len(HORIZONS)))
    daily_qa = []
    schema_metadata = {b'schema_version': b'multihorizon_v1', b'splits_hash': split_hash.encode()}
    writer = None
    try:
        for day, date in enumerate(dates):
            begin, end = day * DAY, (day + 1) * DAY
            stop = min(count, end + 300)
            book = np.asarray(market[begin:stop])
            valid = (book[:, 8] == 1) & (book[:, 7] <= config['quote_max_age_ms'] * 1e6)
            segment = book[:, 9].copy()
            # 区段编号在午夜重建，连续性用原始盘口时刻与有效性判断，来源切换显式隔离。
            gap = ~valid
            gap[1:] |= np.diff(book[:, 5]) < 0
            absolute_seconds = np.arange(begin, stop)
            changed_segment = np.r_[False, np.diff(book[:, 9]) != 0]
            changed_segment &= absolute_seconds % DAY != 0
            gap |= changed_segment
            switch = int((np.datetime64('2025-10-01') - np.datetime64(dates[0])) / np.timedelta64(1, 'D')) * DAY
            gap |= absolute_seconds == switch
            segment = np.cumsum(gap)
            labels = cumulative_labels(book[:, 0], valid & ~gap, segment)[:DAY]
            split = next(name for name, bounds in splits['segments'].items() if bounds['start_row'] <= begin < bounds['stop_row'])
            boundary = splits['segments'][split]['stop_row']
            for index, horizon in enumerate(HORIZONS):
                labels[np.arange(begin, end) + horizon >= boundary, index] = np.nan
            targets[begin:end] = labels
            increments = np.diff(np.column_stack([np.zeros(DAY), labels]), axis=1)
            current_book = book[:DAY]
            features = np.asarray(matrix[begin:end])
            table = pl.DataFrame({
                'decision_ts_ns': timestamps[begin:end], 'book_event_ts_ns': current_book[:, 5].astype(np.int64),
                'book_recv_ts_ns': current_book[:, 6].astype(np.int64),
                'feature_ready_ts_ns': timestamps[begin:end] - SECOND + config['prediction_latency_ns'],
                'mid': current_book[:, 0], 'quote_age_ms': current_book[:, 7] / 1e6,
                'feature_vector': features, 'y_curve_bps': labels, 'g_curve_bps': increments,
                'valid_mask': np.isfinite(labels), 'split': [split] * DAY,
            })
            arrow = table.to_arrow().replace_schema_metadata(schema_metadata)
            if writer is None:
                writer = pq.ParquetWriter(folder / 'dataset.parquet', arrow.schema, compression='zstd')
            writer.write_table(arrow)
            valid_all = np.isfinite(labels).all(axis=1)
            daily_qa.append({'date': date, 'split': split, 'rows': DAY, 'common_valid_rows': int(valid_all.sum()),
                             'invalid_book_rows': int((~valid[:DAY]).sum()),
                             'missing_features_fraction': float((~np.isfinite(features)).mean())})
        targets.flush()
    finally:
        if writer is not None:
            writer.close()
    rows_folder = folder / 'rows'
    rows_folder.mkdir(exist_ok=True)
    split_counts = {}
    for name, bounds in splits['segments'].items():
        positions = np.arange(bounds['start_row'], bounds['stop_row'] - 300, dtype=np.int64)
        positions = positions[np.isfinite(targets[positions]).all(axis=1)]
        if name == 'train':
            positions = positions[positions >= 120]
        np.save(rows_folder / f'{name}.npy', positions)
        split_counts[name] = len(positions)
    manifest = {'feature_names': names, 'feature_locations': locations, 'market_columns': MARKET_COLUMNS,
                'dates': dates, 'rows': count, 'start_timestamp_ns': int(timestamps[0]),
                'horizons_s': HORIZONS.tolist(), 'target_unit': 'log_return_bps',
                'market_latency_ns': 20_000_000, 'feature_latency_ns': config['prediction_latency_ns'],
                'feature_alignment': 'previous_completed_UTC_second', 'splits_hash': split_hash,
                'feature_hash': definition_digest({'locations': locations, 'lineage': lineage}),
                'data_identity': reference, 'clock_status': 'explicit_latency_scenario',
                'lineage_status': 'exact_expression_dependencies_conservative_raw_source_scope'}
    write_json_atomic(folder / 'manifest.json', manifest)
    write_json_atomic(folder / 'data_quality.json', {'daily': daily_qa, 'splits': splits, 'effective_rows': split_counts,
                       'quote_age_effective_max_ms': 1000, 'future_feature_rows': 0,
                       'lineage_resolution': manifest['lineage_status'], 'historical_test_previously_seen': True})
    (folder / 'qa_report.md').write_text('# E0数据验收｜2026-11-09\n\n'
        '正式Features采用20ms行情延迟，策略使用上一完整秒特征并另计1ms计算延迟。'
        '上游盘口有效阈值已为1000ms，维持更严格的现有口径。'
        '16期限采用Float64对数bps，跨缺口、来源切换、分区边界和日尾标签置空。'
        '候选按构造组均匀固定，不使用旧测试IC。表达式依赖逐列记录，原始字段为保守来源范围。'
        '60/20/20区间重新训练，历史Test已被前期研究接触，按历史回看报告。\n')
    print('E0_COMPLETE', count, len(names), flush=True)
