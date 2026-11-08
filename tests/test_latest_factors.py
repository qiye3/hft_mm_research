"""Gen031—037的盘口中心、信息量、历史斜率及完整UTC秒结构独立验证。"""

import datetime as dt
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from mm_research.data.storage import write_parquet
from mm_research.factors.dataset import compute_factor_chunk, factor_module
from mm_research.factors.gen034 import source_expressions as information_sources
from mm_research.factors.gen037 import source_expressions


class LatestFactorTests(unittest.TestCase):
    """指标先按数学定义验证，再核对正式分块入口的整秒窗口。"""

    # 【非核心】逐组检验36列与Float64，预计算源和最终公式分别执行。
    def test_formula_types(self):
        """七个组仅依赖已有基础列，特殊组37的外层输入为整秒。"""
        for number in range(31, 38):
            batch = f'gen{number:03d}'
            module = factor_module(batch)
            config = {'tick_size': .1, 'min_fraction': .8}
            expressions = module.factor_expressions(config)
            inputs = module.source_expressions(config) if number in [31, 34, 37] else expressions
            columns = sorted({name for expression in inputs.values() for name in expression.meta.root_names()})
            data = {}
            for name in columns:
                values = 2 + np.sin(np.arange(1200) / 13)
                if name == 'timestamp_ns':
                    values = np.arange(1200, dtype=np.int64) * 100_000_000
                elif name == 'date':
                    values = ['2026-07-25'] * 1200
                elif name == 'book_segment_id':
                    values = np.zeros(1200, dtype=np.int64)
                elif 'price' in name:
                    values = values + 100000
                data[name] = values
            if number == 37:
                data['timestamp_ns'] = np.arange(1200, dtype=np.int64) * 100_000_000
            table = pl.DataFrame(data)
            if number in [31, 34, 37]:
                table = table.with_columns([expression.alias(name) for name, expression in inputs.items()])
            if number == 37:
                table = table.filter(pl.col('timestamp_ns') % 1_000_000_000 == 0)
            actual = table.select([expression.alias(name) for name, expression in expressions.items()])
            self.assertEqual(actual.width, 36)
            self.assertTrue(all(dtype == pl.Float64 for dtype in actual.schema.values()))

    # 【非核心】交替二元状态有1bit熵、零一阶条件熵和1bit二阶互信息。
    def test_information_alternating_states(self):
        table = pl.DataFrame({'buy_volume': [0., 1.] * 100, 'sell_volume': [1., 0.] * 100,
                              'timestamp_ns': np.arange(200, dtype=np.int64) * 100_000_000,
                              'date': ['2026-07-25'] * 200, 'book_segment_id': [0] * 200,
                              'mid_price': [100., 101.] * 100, 'imbalance_5': [-1., 1.] * 100})
        inputs = information_sources({})
        formulas = factor_module('gen034').factor_expressions({'min_fraction': .8})
        selected = ['info_trade_aggr_sign_100ms_cond_entropy_lag1_5s', 'info_trade_aggr_sign_100ms_mi_lag2_5s']
        derived = table.with_columns([expression.alias(name) for name, expression in inputs.items()])
        actual = derived.select([formulas[name].alias(name) for name in selected])
        self.assertAlmostEqual(actual[selected[0]][-1], 0, places=12)
        self.assertAlmostEqual(actual[selected[1]][-1], 1, places=12)
        prefix = table.head(110).with_columns([expression.alias(name) for name, expression in inputs.items()])
        prefix = prefix.select([formulas[name].alias(name) for name in selected])
        np.testing.assert_allclose(prefix.to_numpy(), actual.head(110).to_numpy(), atol=1e-12, equal_nan=True)

    # 【非核心】稀疏缺失和三种状态的熵、配对边界与独立NumPy经验计数一致。
    def test_information_counts_match_numpy(self):
        """100秒窗口只纳入窗口内的完整状态对，空值保持原时间位置。"""
        random = np.random.default_rng(17)
        states = random.choice([-1., 0., 1., np.nan], size=1500, p=[.3, .3, .35, .05])
        table = pl.DataFrame({'buy_volume': states.tolist(), 'sell_volume': np.zeros(1500),
                              'timestamp_ns': np.arange(1500, dtype=np.int64) * 100_000_000,
                              'date': ['2026-07-25'] * 1500, 'book_segment_id': [0] * 1500,
                              'mid_price': np.ones(1500), 'imbalance_5': np.zeros(1500)}).fill_nan(None)
        derived = table.with_columns([expression.alias(name) for name, expression in information_sources({}).items()])
        formulas = factor_module('gen034').factor_expressions({'min_fraction': .8})
        actual = derived.select([expression.alias(name) for name, expression in formulas.items()]).tail(1)
        window = states[-1000:]
        counts = np.array([(window == value).sum() for value in [-1, 0, 1]])
        probabilities = counts[counts > 0] / counts.sum()
        entropy = -(probabilities * np.log2(probabilities)).sum()
        self.assertAlmostEqual(actual['info_trade_aggr_sign_100ms_entropy_100s'].item(), entropy, places=12)
        for lag in [1, 2]:
            valid = np.array([np.isfinite(window[index - lag:index + 1]).all() for index in range(lag, len(window))])
            old, current = window[:-lag][valid], window[lag:][valid]
            pairs = np.array([((old == previous) & (current == present)).sum()
                              for previous in [-1, 0, 1] for present in [-1, 0, 1]]).reshape(3, 3)
            probabilities = pairs[pairs > 0] / pairs.sum()
            joint = -(probabilities * np.log2(probabilities)).sum()
            row_probabilities = pairs.sum(axis=1) / pairs.sum()
            row_probabilities = row_probabilities[row_probabilities > 0]
            row_entropy = -(row_probabilities * np.log2(row_probabilities)).sum()
            if lag == 1:
                expected = joint - row_entropy
                name = 'info_trade_aggr_sign_100ms_cond_entropy_lag1_100s'
            else:
                column_probabilities = pairs.sum(axis=0) / pairs.sum()
                column_probabilities = column_probabilities[column_probabilities > 0]
                column_entropy = -(column_probabilities * np.log2(column_probabilities)).sum()
                expected = row_entropy + column_entropy - joint
                name = 'info_trade_aggr_sign_100ms_mi_lag2_100s'
            self.assertAlmostEqual(actual[name].item(), expected, places=12)

    # 【非核心】历史收益明确设为上一桶压力的线性函数，局部斜率应为2。
    def test_historical_response_slope(self):
        values = np.sin(np.arange(1200) / 7)
        response = np.concatenate(([0.], values[:-1] * 2 + .3))
        table = pl.DataFrame({'timestamp_ns': np.arange(1200, dtype=np.int64) * 100_000_000,
                              'date': ['2026-07-25'] * 1200, 'book_segment_id': [0] * 1200,
                              'imbalance_5': values, 'trade_imbalance_100ms': values,
                              'trade_imbalance_1s': values, 'return_100ms_bps': response})
        expressions = factor_module('gen036').factor_expressions({'min_fraction': .8})
        actual = table.select([expressions[name].alias(name) for name in ['book_obi_l5_lag_beta_5s', 'book_obi_l5_lag_corr_5s', 'book_obi_l5_lag_asym_beta_5s']])
        np.testing.assert_allclose(actual.tail(200).to_numpy(), np.tile([2., 1., 0.], (200, 1)), atol=1e-10)

    # 【非核心】均匀秒内活动集中度为0.1，含未知位置的零活动秒不能当作已知零。
    def test_intrasecond_known_zero_and_missing(self):
        values = [1.] * 10 + [0.] * 10 + [0.] * 9 + [None]
        table = pl.DataFrame({'timestamp_ns': np.arange(1, 31, dtype=np.int64) * 100_000_000,
                              'trade_count': values, 'buy_volume': values, 'sell_volume': values,
                              'return_100ms_bps': values})
        expressions = source_expressions({})
        actual = table.with_columns([expression.alias(name) for name, expression in expressions.items()]).filter(pl.col('timestamp_ns') % 1_000_000_000 == 0)
        self.assertEqual(actual['intrasec_count_hhi_1s'].to_list(), [.1, 0., None])
        self.assertAlmostEqual(actual['intrasec_count_centroid_1s'][0], 0, places=12)

    # 【非核心】正式分块入口先整合100ms再筛整秒，午夜后5s窗口仍包含前日秒状态。
    def test_intrasecond_chunk_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            dates = ['2026-07-11', '2026-07-12']
            midnight = int(dt.datetime(2026, 7, 12, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
            timestamps = midnight - 120_000_000_000 + np.arange(2400, dtype=np.int64) * 100_000_000
            dataset = {'dates': dates, 'interval_ns': 100_000_000}
            keys = pl.DataFrame({'symbol': ['BTCUSDT'] * 2400, 'date': [dates[0]] * 1200 + [dates[1]] * 1200,
                                 'timestamp_ns': timestamps})
            for name in ['trade_count', 'buy_volume', 'sell_volume', 'return_100ms_bps']:
                table = keys.with_columns(pl.lit(1.).alias(name))
                write_parquet(table.lazy(), folder / f'{name}.parquet', {'kind': 'feature', 'dataset': dataset, 'definition': {}})
            name = 'intrasec_count_hhi_1s_sec_mean_5s'
            compute_factor_chunk({'config': {'tick_size': .1, 'min_fraction': .8, 'write_threads': 1, 'output_interval_ns': 1_000_000_000},
                                  'features': str(folder), 'dates': [dates[1]], 'batch': 'gen037', 'names': [name],
                                  'staging': str(folder / 'staging'), 'index': 0,
                                  'metadata': {name: {'kind': 'factor', 'dataset': dataset, 'definition': {}}}})
            actual = pl.read_parquet(folder / f'staging/0000/{name}.parquet')
            self.assertEqual(actual.height, 120)
            np.testing.assert_allclose(actual[name].to_numpy(), .1, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
