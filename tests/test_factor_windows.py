"""四档窗口的公式数量、600秒跨午夜历史及分块恢复验证。"""

import datetime as dt
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from mm_research.data.storage import write_parquet
from mm_research.factors.dataset import EXTENDED_BATCHES, SUPPORTED_BATCHES, compute_factor_chunk, factor_module
from mm_research.evaluation.metrics import single_factor_ols


class FactorWindowTests(unittest.TestCase):
    # 【非核心】验证区间均值与训练不同，OLS残差仍使用对应区间的自变量统计。
    def test_ols_shifted_validation_distribution(self):
        """已知y=2x+7，在训练与验证不同取值范围下均应精确恢复。"""
        statistics = {}
        for split, values in [('train', np.arange(20.)), ('val_dev', np.arange(100., 110.))]:
            targets = 2 * values + 7
            statistics[split] = [len(values), values.sum(), values @ values,
                                 targets.sum(), targets @ targets, values @ targets]
        self.assertAlmostEqual(single_factor_ols(statistics)['val_dev_oos_r2'], 1.)

    # 【非核心】四档普通窗口与三对短长窗口生成固定1206个候选，名称全项目唯一。
    def test_formula_inventory(self):
        """检查35个现有公式组的完整展开，避免窗口修改遗漏专用算子。"""
        config = {'tick_size': .1, 'min_fraction': .8, 'aggregation_windows_s': [10, 60, 180, 600],
                  'mean_diff_windows_s': [[10, 60], [60, 180], [180, 600]]}
        names = []
        for batch in (*SUPPORTED_BATCHES, *EXTENDED_BATCHES):
            if batch != 'gen030':
                names.extend(factor_module(batch).factor_expressions(config))
        self.assertEqual(len(names), 1206)
        self.assertEqual(len(set(names)), 1206)

    # 【非核心】600秒盘口墙与熵统计在午夜保留完整历史，已完成分块可以直接恢复。
    def test_long_precomputed_window_at_midnight(self):
        """真实分块入口与连续Polars计算逐行对照，旧120秒尾部会使本例失效。"""
        midnight = int(dt.datetime(2026, 7, 12, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
        timestamps = midnight - 700_000_000_000 + np.arange(14000, dtype=np.int64) * 100_000_000
        config = {'tick_size': .1, 'min_fraction': .8, 'aggregation_windows_s': [600],
                  'mean_diff_windows_s': [[180, 600]], 'write_threads': 1,
                  'output_interval_ns': 1_000_000_000, 'dates': ['2026-07-12']}
        source = pl.DataFrame({'symbol': ['BTCUSDT'] * len(timestamps),
                               'date': ['2026-07-11'] * 7000 + ['2026-07-12'] * 7000,
                               'timestamp_ns': timestamps,
                               'book_segment_id': [0] * len(timestamps),
                               'mid_price': 60000 + np.sin(np.arange(len(timestamps)) / 30),
                               'imbalance_5': np.sin(np.arange(len(timestamps)) / 20),
                               'buy_volume': np.ones(len(timestamps)),
                               'sell_volume': np.zeros(len(timestamps))})
        for level in range(1, 11):
            source = source.with_columns(pl.lit(60000. - level).alias(f'bid_price_{level}'),
                                         pl.lit(60000. + level).alias(f'ask_price_{level}'),
                                         pl.lit(float(level)).alias(f'bid_quantity_{level}'),
                                         pl.lit(float(11 - level)).alias(f'ask_quantity_{level}'))
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            dataset = {'dates': ['2026-07-11', '2026-07-12'], 'interval_ns': 100_000_000}
            for name in source.columns[3:]:
                write_parquet(source.select('symbol', 'date', 'timestamp_ns', name).lazy(),
                              folder / f'{name}.parquet', {'kind': 'feature', 'dataset': dataset, 'definition': {}})
            for batch in ['gen022', 'gen034']:
                module = factor_module(batch)
                formulas = module.factor_expressions(config)
                inner = source.select('symbol', 'date', 'timestamp_ns',
                                      *[expression.alias(name) for name, expression in module.source_expressions(config).items()])
                expected = inner.select('symbol', 'date', 'timestamp_ns',
                                        *[expression.alias(name) for name, expression in formulas.items()])
                expected = expected.filter((pl.col('timestamp_ns') >= midnight)
                                           & (pl.col('timestamp_ns') % 1_000_000_000 == 0))
                task = {'config': config, 'features': str(folder), 'dates': ['2026-07-12'],
                        'batch': batch, 'names': list(formulas), 'staging': str(folder / batch), 'index': 0,
                        'metadata': {name: {'kind': 'factor', 'dataset': dataset, 'definition': {}} for name in formulas}}
                usage = compute_factor_chunk(task)
                actual = None
                for name in formulas:
                    table = pl.read_parquet(folder / batch / '0000' / f'{name}.parquet')
                    actual = table if actual is None else actual.with_columns(table[name])
                assert_frame_equal(actual, expected, check_exact=False, abs_tol=1e-10)
                self.assertEqual(compute_factor_chunk(task)['wall_seconds'], usage['wall_seconds'])


if __name__ == '__main__':
    unittest.main()
