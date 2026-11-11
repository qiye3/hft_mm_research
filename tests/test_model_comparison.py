"""正式模型对照的训练期选择、缺失处理、跨日历史与配对统计回归。"""

import datetime as dt
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from mm_research.data.storage import write_parquet
from mm_research.factors.dataset import factor_module
from mm_research.models.comparison_contract import select_comparison_inputs, comparison_derivations, comparison_indices
from mm_research.models.comparison_metrics import paired_day_comparison
from mm_research.models.dataset import RAW_COLUMNS
from mm_research.models.experiment_data import HORIZONS, prepare_experiment_data, history_batch


class ComparisonTests(unittest.TestCase):
    """同一数据契约禁止验证或测试标签参与选列，窗口保留跨日时间位置。"""

    # 【非核心】改变验证和测试IC不能改变候选，固定覆盖门槛只读取训练统计。
    def test_selection_ignores_validation_and_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {'train_end': '2026-05-31', 'start_date': '2025-05-09', 'minimum_train_coverage': .95,
                      'feature_folder': 'features', 'families': {'price': [5], 'obi': [1, 2, 19], 'depth': [6]},
                      'factor_sources': [{'folder': 'factors', 'evaluation': 'evaluation',
                                          'batches': ['gen001', 'gen002', 'gen005', 'gen006', 'gen019']}]}
            filenames = []
            for batch in config['factor_sources'][0]['batches']:
                for index, name in enumerate(factor_module(batch).factor_expressions({'min_fraction': .8, 'tick_size': .1})):
                    filename = root / 'evaluation' / name / 'summary.parquet'
                    filename.parent.mkdir(parents=True)
                    rows = [{'factor': name, 'split': split, 'sampling': 'all', 'horizon_ns': 1_000_000_000,
                             'ic_mean': (index + 1) / 1000 if split == 'train' else 1000.,
                             'valid_ic_days': 388, 'paired_rows': 388 * 86400} for split in ['train', 'validation', 'test']]
                    pl.DataFrame(rows).write_parquet(filename)
                    filenames.append(filename)
            before = select_comparison_inputs(root, config)
            for filename in filenames:
                table = pl.read_parquet(filename).with_columns(pl.when(pl.col('split') == 'train').then(pl.col('ic_mean')).otherwise(-9999.).alias('ic_mean'))
                table.write_parquet(filename)
            after = select_comparison_inputs(root, config)
            self.assertEqual(before, after)
            self.assertEqual(len(before['groups']['base90']), 90)
            self.assertFalse(before['selection_uses_test'])

    # 【非核心】追加未来极端值不会改变已有派生值，窗口不按日期清空。
    def test_derivation_causality_and_missing(self):
        table = pl.DataFrame({'value': [float(index) for index in range(240)], 'bid': [0.] * 240, 'ask': [0.] * 240})
        definitions = {'lag': {'kind': 'lag', 'input': 'value', 'seconds': 30},
                       'deviation': {'kind': 'deviation', 'input': 'value', 'seconds': 120},
                       'ratio': {'kind': 'ratio', 'inputs': ['bid', 'ask']}}
        expressions = comparison_derivations(definitions)
        full = table.with_columns(expressions)
        assert_frame_equal(table.head(180).with_columns(expressions), full.head(180))
        self.assertEqual(full['lag'][120], 90.)
        self.assertAlmostEqual(full['deviation'][120], 59.5)
        self.assertEqual(full['ratio'][120], 0.)

    # 【非核心】先对齐每日差值再重采样，固定差值的配对区间必须退化到该常数。
    def test_paired_bootstrap_uses_same_dates(self):
        left = [{'day': day, 'horizon_seconds': 1, 'ic': day / 100 + .05, 'mse_bps2': 1.} for day in range(51)]
        right = [{'day': day, 'horizon_seconds': 1, 'ic': day / 100, 'mse_bps2': 1.02} for day in range(51)]
        result = paired_day_comparison(left, right, 7)
        np.testing.assert_allclose(result['ci95'], [.05, .05], atol=1e-15)
        np.testing.assert_allclose(result['mse_improvement_ci95'], [.02, .02], atol=1e-15)

    # 【非核心】真实缓存入口跨午夜携带滞后值，并按目标时间和300秒边界屏蔽标签。
    def test_disk_cache_derivation_and_common_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates = ['2026-05-31', '2026-06-01', '2026-06-02']
            start = int(dt.datetime(2026, 5, 31, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
            timestamps = start + np.arange(3 * 86400, dtype=np.int64) * 1_000_000_000
            values = np.sin(np.arange(len(timestamps)) / 19)
            keys = pl.DataFrame({'symbol': ['BTCUSDT'] * len(values), 'date': [date for date in dates for _ in range(86400)], 'timestamp_ns': timestamps})
            dataset = {'dates': dates, 'symbol': 'BTCUSDT', 'interval_ns': 1_000_000_000}
            inputs = {name: 'features' for name in RAW_COLUMNS}
            for name in inputs:
                write_parquet(keys.with_columns(pl.Series(name, values)).lazy(), root / 'features' / f'{name}.parquet',
                              {'kind': 'feature', 'dataset': dataset, 'definition': {'group': 'fixture'}})
            for date in dates:
                daily = keys.filter(pl.col('date') == date)
                for seconds in HORIZONS:
                    end = pl.col('timestamp_ns') + seconds * 1_000_000_000
                    valid = end < start + len(values) * 1_000_000_000
                    labels = daily.with_columns(valid.alias('label_valid'), pl.when(valid).then(end).alias('label_end_timestamp_ns'),
                                                pl.when(valid).then(pl.col('timestamp_ns').cast(pl.Float64) / 1e12).alias(f'y_mid_{seconds}s_bps'))
                    write_parquet(labels.lazy(), root / 'labels' / f'horizon={seconds}s' / f'{date}.parquet',
                                  {'kind': 'labels', 'dataset': dataset, 'horizon_ns': seconds * 1_000_000_000})
            config = {'dataset_output': 'cache', 'label_folder': 'labels', 'train_end': dates[0], 'validation_end': dates[1],
                      'training_rows': 1000, 'validation_rows': 100, 'seed': 7, 'read_workers': 2}
            definitions = {'lag': {'kind': 'lag', 'input': RAW_COLUMNS[0], 'seconds': 30}}
            metadata = prepare_experiment_data(root, config, inputs, definitions)
            matrix = np.load(root / 'cache/x.npy', mmap_mode='r')
            targets = np.load(root / 'cache/y.npy', mmap_mode='r')
            sequential_config = {**config, 'dataset_output': 'cache_sequential', 'read_workers': 1}
            sequential_metadata = prepare_experiment_data(root, sequential_config, inputs, definitions)
            np.testing.assert_array_equal(matrix, np.load(root / 'cache_sequential/x.npy'))
            self.assertEqual(metadata['mean_x'], sequential_metadata['mean_x'])
            column = metadata['columns'].index('lag')
            restored = matrix[86400, column] * metadata['scale_x'][column] + metadata['mean_x'][column]
            self.assertAlmostEqual(restored, values[86400 - 30], places=6)
            self.assertTrue(np.isnan(targets[86400 - 300:86400]).all())
            self.assertTrue(np.isnan(targets[2 * 86400 - 300:2 * 86400]).all())
            rows = comparison_indices(metadata, targets, config)
            self.assertGreaterEqual(rows['fit'].min(), 120)
            changed = targets.copy()
            changed[metadata['test_start']:] *= 100
            np.testing.assert_array_equal(rows['fit'], comparison_indices(metadata, changed, config)['fit'])
            probe = np.array([[np.nan, 1.]], dtype=np.float32)
            batch = history_batch(probe, np.array([0]), np.array([0, 1]), 1)
            np.testing.assert_array_equal(batch, [[[0., 1., 1., 0.]]])


if __name__ == '__main__':
    unittest.main()
