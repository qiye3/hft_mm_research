"""Gen021—030的方向记忆、状态确认及新增逐笔分布数值检查。"""

import unittest
import numpy as np
import polars as pl

from mm_research.factors.dataset import factor_module
from mm_research.factors.event_states import breakout_states
from mm_research.factors.gen021 import source_expressions
from mm_research.factors.gen025 import shock_states
from mm_research.features.trade_distribution import distribution_bins


class MoreFactorTests(unittest.TestCase):
    """有状态因子只在实际事件发生时更新，数据缺失不当作有效零。"""

    # 【非核心】逐组检查完整命名及浮点输出，Gen030只验证公式不生成现货数据。
    def test_formula_types(self):
        """包含状态源预处理和有界失败突破内核，所有组各36个。"""
        for number in range(21, 31):
            batch = f'gen{number:03d}'
            expressions = factor_module(batch).factor_expressions({'min_fraction': .8, 'tick_size': .1})
            inputs = factor_module(batch).source_expressions({'tick_size': .1, 'min_fraction': .8}) if number in [21, 22, 25, 26, 28] else expressions
            names = sorted({name for expression in inputs.values() for name in expression.meta.root_names()})
            data = {}
            for name in names:
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
            table = pl.DataFrame(data)
            if number in [21, 22, 25, 26, 28]:
                table = table.with_columns([expression.alias(name) for name, expression in inputs.items()])
            result = table.select([expression.alias(name) for name, expression in expressions.items()])
            self.assertEqual(result.width, 36)
            self.assertTrue(all(dtype == pl.Float64 for dtype in result.schema.values()))

    # 【非核心】相等买卖量中断方向长度，缺失后的新方向从一开始。
    def test_direction_runs(self):
        table = pl.DataFrame({'timestamp_ns': np.arange(7, dtype=np.int64) * 100_000_000,
                              'buy_volume': [1., 2., 0., 0., 0., None, 1.],
                              'sell_volume': [0., 0., 0., 1., 2., None, 0.]})
        result = table.select(source_expressions({})['trade_dir_signed_run_100ms']).to_series()
        self.assertEqual(result.to_list(), [1., 2., 0., -1., -2., None, 1.])

    # 【非核心】达到80%恢复的当前桶结束冲击，后续不能重新激活旧状态。
    def test_shock_recovery_is_causal(self):
        spread = [1.] * 1005 + [4., 3., 1.5, 3., 3.]
        table = pl.DataFrame({'timestamp_ns': np.arange(len(spread), dtype=np.int64) * 100_000_000,
                              'date': ['2026-07-25'] * len(spread), 'book_segment_id': [0] * len(spread), 'spread': spread})
        expressions = shock_states(pl.col('spread'), True)
        actual = table.select([expression.alias(name) for name, expression in expressions.items()])
        self.assertEqual(actual['active'].tail(5).to_list(), [1., 1., 0., 0., 0.])
        self.assertAlmostEqual(actual['recovery'][1006], 1/3)
        self.assertAlmostEqual(actual['age_sec'][1006], .1)

    # 【非核心】回落在实际确认桶记录一次，确认以前的前缀中没有未来标记。
    def test_failed_breakout_confirmation(self):
        mid = np.array([100.] * 50 + [101.] * 5 + [100.] * 30)
        allowed = np.ones(len(mid), dtype=bool)
        allowed[0] = False
        result = breakout_states(mid, allowed, .1)
        self.assertEqual(result[50, 0], 1)
        self.assertEqual(result[55, 6], 1)
        self.assertEqual(np.nansum(result[:, 6]), 1)
        np.testing.assert_allclose(result[:55], breakout_states(mid[:55], allowed[:55], .1), equal_nan=True)

    # 【非核心】十一笔成交取最大的两笔作为头部，空卖侧保持零。
    def test_trade_distribution_top_decile(self):
        table = pl.DataFrame({'timestamp_ns': [0] * 11, 'side': ['buy'] * 11,
                              'quantity': [float(value) for value in range(1, 12)]})
        result = distribution_bins(table.lazy()).collect().row(0, named=True)
        self.assertEqual(result['trade_size_buy_median_qty_100ms'], 6)
        self.assertAlmostEqual(result['trade_size_buy_topdecile_share_100ms'], 21/66)
        self.assertAlmostEqual(result['trade_size_buy_hhi_100ms'], sum(value**2 for value in range(1, 12))/66**2)
        self.assertEqual(result['trade_size_sell_topdecile_share_100ms'], 0)


if __name__ == '__main__':
    unittest.main()
