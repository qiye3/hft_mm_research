"""Gen011—020文档名称、可执行深度、年龄与历史相关的独立数值检查。"""

import unittest

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from mm_research.factors.dataset import factor_module

EXTENDED_BATCHES = tuple(f"gen{index:03d}" for index in range(11, 21))
from mm_research.factors.gen012 import execution_vwap
from mm_research.factors.gen014 import elapsed_since_event
from mm_research.factors.gen020 import slot_statistics


class ExtendedFactorTests(unittest.TestCase):
    """新增公式使用当前100ms输入，外层每秒取样且不重置缺失窗口。"""

    # 【非核心】逐组生成全部表达式，检查输出Float64与前缀因果性。
    def test_names_types_and_prefix_causality(self):
        """所有组36个，参考列只提供事前常数，不涉及未来数据。"""
        for batch in EXTENDED_BATCHES:
            expressions = factor_module(batch).factor_expressions({"tick_size": 0.1, "min_fraction": 0.8})
            self.assertEqual(len(expressions), 36)
            inputs = factor_module(batch).source_expressions({'tick_size': .1, 'min_fraction': .8}) if batch in ['gen012', 'gen014'] else expressions
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
                elif name == 'flow_valid':
                    values = np.ones(1200, dtype=bool)
                elif 'price' in name:
                    values = values + 100000
                data[name] = values
            table = pl.DataFrame(data)
            if batch in ["gen012", "gen014"]:
                table = table.with_columns([expression.alias(name) for name, expression in inputs.items()])
            formulas = [expression.alias(name) for name, expression in expressions.items()]
            complete = table.select(formulas)
            self.assertTrue(all(dtype == pl.Float64 for dtype in complete.schema.values()))
            assert_frame_equal(table.head(1100).select(formulas), complete.head(1100), check_exact=False, abs_tol=1e-8)

    # 【非核心】独立手算末档部分吃单，深度不足不得外推均价。
    def test_execution_price_and_insufficient_depth(self):
        """2.5BTC在101买1BTC、102买1.5BTC，VWAP为101.6。"""
        data = {f'ask_price_{level}': [100.0 + level] for level in range(1, 11)}
        data.update({f'ask_quantity_{level}': [0.0] for level in range(1, 11)})
        data['ask_quantity_1'] = [1.0]
        data['ask_quantity_2'] = [2.0]
        table = pl.DataFrame(data)
        self.assertAlmostEqual(table.select(execution_vwap('ask', pl.lit(2.5))).item(), 101.6)
        self.assertIsNone(table.select(execution_vwap('ask', pl.lit(3.1))).item())

    # 【非核心】缺失后须重新观察变化事件，不能沿用前区段年龄。
    def test_age_initialization_and_missing(self):
        """连续不变时按秒递增，首个变化前和恢复初始化阶段为空。"""
        table = pl.DataFrame({'timestamp_ns': np.arange(7, dtype=np.int64) * 100_000_000,
                              'quote': [1.0, 2.0, 2.0, None, 2.0, 3.0, 3.0]})
        quote = pl.col('quote')
        observed = table.select(elapsed_since_event(quote.is_not_null(), quote != quote.shift(1))).to_series().to_list()
        self.assertEqual(observed, [None, 0.0, 0.1, None, None, 0.0, 0.1])

    # 【非核心】同槽参考用原始样本的合并分布，缺失只影响对应时段变量。
    def test_slot_reference_statistics(self):
        """均值、ddof=1标准差和线性Q90均与独立NumPy切片一致。"""
        past = np.arange(20 * 2 * 3 * 2, dtype=float).reshape(20, 2, 3, 2)
        past[0, 1, 0, 1] = np.nan
        result = slot_statistics(past, '2026-07-25', ['first', 'second'])
        values = past[:, 0, :, 0].reshape(-1)
        self.assertAlmostEqual(result[0]['_slot_first_mean'], float(np.mean(values)))
        self.assertAlmostEqual(result[0]['_slot_first_std'], float(np.std(values, ddof=1)))
        self.assertAlmostEqual(result[0]['_slot_first_q90'], float(np.quantile(values, .9)))
        self.assertIsNone(result[1]['_slot_second_mean'])
        self.assertIsNotNone(result[1]['_slot_first_mean'])


if __name__ == '__main__':
    unittest.main()
