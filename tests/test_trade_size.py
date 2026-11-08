"""大小单分桶必须保持100ms网格顺序，供后续滚动窗口直接使用。"""

import datetime as dt
from pathlib import Path
import unittest
from unittest.mock import patch

import polars as pl

from mm_research.features.trade_size import build_size_day


class TradeSizeTests(unittest.TestCase):
    """核对乱序成交聚合后的时间顺序及已知零成交口径。"""

    # 【非核心】成交分桶关联必须保留左侧网格顺序，未知流量保持null。
    def test_size_bins_preserve_grid_order(self):
        """倒序成交输入不应改变输出网格，边界样本按20ms延迟归桶。"""
        date = "2026-07-25"
        start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
        grid = pl.DataFrame({
            "symbol": ["BTCUSDT"] * 4,
            "date": [date] * 4,
            "timestamp_ns": [start + offset * 100_000_000 for offset in range(4)],
            "flow_valid": [True, True, True, False],
        })
        trades = pl.DataFrame({
            "exchange_timestamp_ns": [start + 150_000_000, start + 50_000_000],
            "quantity": [0.1, 0.002],
            "side": ["sell", "buy"],
            "quality_flags": [[], []],
        }, schema_overrides={"quality_flags": pl.List(pl.String)})
        dataset = {"market_data_latency_ns": 20_000_000, "interval_ns": 100_000_000, "lot_size": 0.001}
        with patch("mm_research.features.trade_size.scan_standard_history", return_value=trades.lazy()), patch(
            "mm_research.features.trade_size.scan_series", return_value=(grid.lazy(), dataset),
        ):
            result = build_size_day((Path("standard"), Path("features"), date, dataset, {"q50": 0.003, "q90": 0.08}))
        self.assertEqual(result["timestamp_ns"].to_list(), grid["timestamp_ns"].to_list())
        self.assertEqual(result["trade_small_buy_qty_100ms"].to_list(), [0.0, 0.002, 0.0, None])
        self.assertEqual(result["trade_large_sell_qty_100ms"].to_list(), [0.0, 0.0, 0.1, None])


if __name__ == "__main__":
    unittest.main()
