"""验证独立来源特征模块示例的公式和因果边界。

人工盘口与成交序列检查金额、窗口、缺口和公共运算。
"""

import unittest

import polars as pl
from polars.testing import assert_frame_equal

from mm_research.features.expressions import normalized_imbalance, rolling_sum_in_segment
from mm_research.features.snapshot import SNAPSHOT_EXAMPLE_COLUMNS, add_snapshot_notional_features
from mm_research.features.trade_events import build_trade_event_features

class FeatureExampleTests(unittest.TestCase):
    """检查 Polars 模块示例的数据口径及共用数学运算。"""

    # 盘口金额公式可手算，失效盘口不能输出可用特征。
    def test_snapshot_notional_features(self):
        """输入两行相同盘口和不同质量标记，检查金额单位、失衡和空值。"""
        values = {"book_valid": [True, False]}
        for level in range(1, 11):
            values[f"bid_price_{level}"] = [100.0 - level] * 2
            values[f"ask_price_{level}"] = [100.0 + level] * 2
            values[f"bid_quantity_{level}"] = [2.0] * 2
            values[f"ask_quantity_{level}"] = [1.0] * 2
        table = add_snapshot_notional_features(pl.DataFrame(values))
        self.assertEqual(table["snapshot_bid_notional_10"][0], 1890.0)
        self.assertEqual(table["snapshot_ask_notional_10"][0], 1055.0)
        self.assertAlmostEqual(table["snapshot_notional_imbalance_10"][0], 835 / 2945)
        for column in SNAPSHOT_EXAMPLE_COLUMNS:
            self.assertIsNone(table[column][1])

    # 净额、笔数和跳号后的重预热都只使用采样时刻已收到的成交。
    def test_trade_windows_gaps_and_causality(self):
        """输入已分桶成交和 100ms 网格，核对窗口右端点、缺口和追加未来数据。"""
        grid = pl.DataFrame({"timestamp_ns": [index * 100_000_000 for index in range(150)]})
        bins = pl.DataFrame({
            "timestamp_ns": [5_100_000_000, 5_200_000_000, 7_000_000_000],
            "trade_count": [2, 1, 1],
            "buy_notional": [20.0, 0.0, 10.0],
            "sell_notional": [0.0, 5.0, 0.0],
            "trade_sequence_gap": [False, False, True],
        })
        table = build_trade_event_features(bins, grid)
        self.assertEqual(table["trade_net_notional_1s"][52], 15.0)
        self.assertEqual(table["trade_intensity_1s"][52], 3.0)
        self.assertEqual(table["trade_net_notional_100ms"][52], -5.0)
        self.assertEqual(table["trade_intensity_1s"][62], 0.0)
        self.assertFalse(table["trade_feature_valid"][49])
        self.assertTrue(table["trade_feature_valid"][50])
        self.assertFalse(table["trade_feature_valid"][70])
        self.assertFalse(table["trade_feature_valid"][119])
        self.assertTrue(table["trade_feature_valid"][120])
        self.assertIsNone(table["trade_net_notional_1s"][70])
        prefix = build_trade_event_features(bins.head(2), grid.head(60))
        assert_frame_equal(prefix, table.head(60), check_exact=True)

    # 公共运算区分未知值与零成交，整数窗口不跨质量区段。
    def test_shared_expression_boundaries(self):
        """输入零量、缺失量及两个区段，检查空窗约定与窗口重新预热。"""
        table = pl.DataFrame({
            "buy": [0.0, None, 0.002, 0.002],
            "sell": [0.0, 0.001, 0.001, 0.001],
            "segment_id": [0, 0, 1, 1],
        }).with_columns(
            normalized_imbalance(pl.col("buy"), pl.col("sell"), empty_value=0.0).alias("imbalance"),
            rolling_sum_in_segment(pl.col("buy"), 2, quantity_step=0.001).alias("volume"),
        )
        self.assertEqual(table["imbalance"][0], 0.0)
        self.assertIsNone(table["imbalance"][1])
        self.assertIsNone(table["volume"][2])
        self.assertEqual(table["volume"][3], 0.004)

if __name__ == "__main__":
    unittest.main()
