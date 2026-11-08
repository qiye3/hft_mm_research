"""独立验证共享模型矩阵的顺序NPY写入、训练期标准化与300秒分界隔离。"""
import datetime as dt
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from mm_research.data.storage import write_parquet
from mm_research.models.dataset import RAW_COLUMNS
from mm_research.models.experiment_data import HORIZONS, prepare_experiment_data


class ExperimentDataTests(unittest.TestCase):
    """使用完整三日100ms夹具，经真实入口生成整秒缓存。"""

    # 【非核心】缓存写入后能被标准NumPy读取，均值和标签屏蔽与独立数组一致。
    def test_sequential_matrix_and_partition_purge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates = ['2026-05-31', '2026-06-01', '2026-06-02']
            start = int(dt.datetime(2026, 5, 31, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
            timestamps = start + np.arange(3 * 864000, dtype=np.int64) * 100_000_000
            keys = pl.DataFrame({'symbol': ['BTCUSDT'] * len(timestamps),
                                 'date': [date for date in dates for _ in range(864000)], 'timestamp_ns': timestamps})
            values = np.sin(np.arange(len(timestamps)) % 1000 / 13)
            dataset = {'dates': dates, 'symbol': 'BTCUSDT', 'interval_ns': 100_000_000}
            for name in RAW_COLUMNS:
                table = keys.with_columns(pl.Series(name, values))
                write_parquet(table.lazy(), root / 'features' / f'{name}.parquet',
                              {'kind': 'feature', 'dataset': dataset, 'definition': {'group': 'fixture'}})
            sampled = keys.filter(pl.col('timestamp_ns') % 1_000_000_000 == 0)
            for date in dates:
                daily = sampled.filter(pl.col('date') == date)
                for seconds in HORIZONS:
                    end = pl.col('timestamp_ns') + seconds * 1_000_000_000
                    valid = end < start + 3 * 86400 * 1_000_000_000
                    labels = daily.with_columns(valid.alias('label_valid'),
                                                pl.when(valid).then(end).alias('label_end_timestamp_ns'),
                                                pl.when(valid).then(pl.lit(.1)).alias(f'y_mid_{seconds}s_bps'))
                    write_parquet(labels.lazy(), root / 'labels' / f'horizon={seconds}s' / f'{date}.parquet',
                                  {'kind': 'labels', 'dataset': dataset, 'horizon_ns': seconds * 1_000_000_000})
            metadata = prepare_experiment_data(root, {
                'feature_folder': 'features', 'factor_sources': [], 'candidates_per_group': 2,
                'skip_unavailable_groups': False, 'label_folder': 'labels', 'dataset_output': 'cache',
                'train_end': dates[0], 'validation_end': dates[1],
            })
            matrix = np.load(root / 'cache/x.npy', mmap_mode='r')
            targets = np.load(root / 'cache/y.npy', mmap_mode='r')
            training = values[::10][:86400 - 300]
            self.assertEqual(matrix.shape, (3 * 86400, 6))
            self.assertEqual(matrix.dtype, np.float32)
            np.testing.assert_allclose(matrix[:100, 0], (values[::10][:100].astype(np.float32) - training.mean()) / training.std(), atol=1e-6)
            self.assertTrue(np.isnan(targets[86400 - 300:86400]).all())
            self.assertTrue(np.isnan(targets[2 * 86400 - 300:2 * 86400]).all())
            self.assertEqual(metadata['train_stop'], 86400 - 300)
            self.assertFalse((root / 'cache/raw_x.npy').exists())


if __name__ == '__main__':
    unittest.main()
