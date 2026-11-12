"""新增跨市场变量的因果窗口测试，覆盖未来扰动与跨日连续计算。"""

import unittest
from pathlib import Path
import tempfile

import numpy as np
import polars as pl

from mm_research.factors.cross_market import cross_market_factors
from mm_research.models.cross_market import publish_cache, feature_group_indices
from mm_research.models.cross_market_ablation import read_ensemble


class CrossMarketFactorsTest(unittest.TestCase):
    # 【非核心】消融必须覆盖完整现货组且互斥，避免把两组差异混入重复列。
    def test_ablation_partition(self) -> None:
        """核对列选择，不依赖远端矩阵。"""
        columns = ['spot_imbalance_10s', 'spot_return_10s', 'spot_close_perp_mid_gap_10s', 'premium_level_bps']
        flow = feature_group_indices(columns, 'flow')
        price = feature_group_indices(columns, 'price')
        self.assertEqual(flow, [0])
        self.assertEqual(price, [1, 2])
        self.assertEqual(sorted(flow + price), feature_group_indices(columns, 'spot'))

    # 【非核心】标签无效的时刻也应输出平均预测，避免按未来标签删掉决策行。
    def test_ensemble_retains_decision_rows(self) -> None:
        """三种子预测取实际均值，评价掩码保留原值。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, prediction in [(17, 1.0), (42, 2.0), (73, 6.0)]:
                folder = root / 'flow/models' / f'180s_combined_xgb_seed{seed}'
                folder.mkdir(parents=True)
                pl.DataFrame({
                    'timestamp_ns': [0, 1], 'prediction_log_bps': [prediction, prediction],
                    'target_log_bps': [2.0, None], 'evaluation_valid': [True, False],
                }).write_parquet(folder / 'val_dev.parquet')
            combined = read_ensemble(root, 'flow', 'val_dev')
            self.assertEqual(combined['prediction_log_bps'].to_list(), [3.0, 3.0])
            self.assertEqual(combined['evaluation_valid'].to_list(), [True, False])

    # 【非核心】本地矩阵顺序发布后仍可只读映射，并释放该矩阵的临时占用。
    def test_cache_publication(self) -> None:
        """验证新增缓存发布路径，不重跑任何模型训练。"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'local.npy'
            destination = Path(directory) / 'shared.npy'
            values = np.arange(12, dtype='float32').reshape(3, 4)
            np.save(source, values)
            publish_cache(source, destination)
            np.testing.assert_array_equal(np.load(destination, mmap_mode='r'), values)
            self.assertFalse(source.exists())

    # 【非核心】未来价格扰动不能改变已有决策时刻的因子，跨午夜不截断滚动。
    def test_future_invariance_and_flow(self) -> None:
        """使用跨午夜的固定买卖量，核对压力定义和前缀一致性。"""
        count = 1500
        frame = pl.DataFrame({
            'timestamp_ns': (np.arange(count) + 86300) * 1_000_000_000,
            'spot_close': 100.0 + np.arange(count) * 0.01,
            'volume': np.full(count, 10.0), 'buy_volume': np.full(count, 7.0),
            'perp_mid': np.full(count, 100.0), 'premium_bps': np.full(count, 2.0),
        })
        original = cross_market_factors(frame, [10, 60, 180, 600])
        changed = frame.with_row_index().with_columns(
            pl.when(pl.col('index') >= 1100).then(99999.0).otherwise(pl.col('spot_close')).alias('spot_close'),
        ).drop('index')
        alternative = cross_market_factors(changed, [10, 60, 180, 600])
        self.assertTrue(original.head(1100).equals(alternative.head(1100)))
        self.assertAlmostEqual(original['spot_imbalance_600s'][1000], 0.4)
        self.assertEqual(original['premium_change_600s'][1000], 0)
        self.assertEqual(len(original.columns), 21)


if __name__ == '__main__':
    unittest.main()
