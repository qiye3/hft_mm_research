"""核对当前九个非OFI组的公式、缺失规则、因果性与跨块连续性。"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from mm_research.data.storage import write_parquet
from mm_research.factors.dataset import SUPPORTED_BATCHES, compute_factor_chunk, factor_module
from mm_research.factors.operators import rolling_factors


class FactorTests(unittest.TestCase):
    """文档本轮共300个非OFI因子，核心窗口由独立NumPy逐行对照。"""

    # 【非核心】逐窗参考核对原生算子、空值、NaN和前缀因果性。
    def test_rolling_reference_and_causality(self):
        """120个连续时间点包含缺失，均值差跨午夜继续滑动。"""
        values = np.sin(np.arange(1200) / 9)
        values[[17, 400, 550]] = np.nan
        table = pl.DataFrame({"pressure": values})
        expressions = rolling_factors({"pressure": pl.col("pressure")}, ((10, 50),), "mean_diff", 0.8)
        expression = expressions["pressure_mean_diff_10s_50s"]
        result = table.select(expression.alias("value"))
        expected = []
        for index in range(len(values)):
            short = values[max(0, index - 99):index + 1]
            long = values[max(0, index - 499):index + 1]
            valid = np.isfinite(short).sum() >= 80 and np.isfinite(long).sum() >= 400
            expected.append(np.nanmean(short) - np.nanmean(long) if valid else np.nan)
        np.testing.assert_allclose(result["value"].to_numpy(), expected, atol=1e-12, rtol=0, equal_nan=True)
        assert_frame_equal(table.head(75).select(expression.alias("value")), result.head(75))

    # 【非核心】核对各组完整因子名、数量与必要公式定义，明确排除OFI。
    def test_documented_groups(self):
        """组001至006和010各36个，008和009各24个。"""
        counts = []
        for batch in SUPPORTED_BATCHES:
            expressions = factor_module(batch).factor_expressions({"min_fraction": 0.8, "tick_size": 0.1, "aggregation_windows_s": ([3, 5, 10, 20, 50, 100] if batch == "gen001" else [5, 10, 20, 50]), "mean_diff_windows_s": [[3, 10], [5, 20], [5, 50], [10, 50], [10, 100], [20, 100]]})
            counts.append(len(expressions))
            inputs = sorted({column for expression in expressions.values() for column in expression.meta.root_names()})
            table = pl.DataFrame({column: [1, 2] if column == "trade_count" else [1.0, 2.0] for column in inputs})
            result = table.select([expression.alias(name) for name, expression in expressions.items()])
            self.assertTrue(all(dtype == pl.Float64 for dtype in result.schema.values()))
            self.assertFalse(any("ofi" in name for name in expressions))
        self.assertEqual(counts, [36, 36, 36, 36, 36, 36, 24, 24, 36])
        self.assertEqual(sum(counts), 300)
        with self.assertRaises(ValueError):
            rolling_factors({"value": pl.col("value")}, (10,), "mean", 1.1)

    # 【非核心】真实文件分块入口在午夜接入前日历史，与全区间计算逐行一致。
    def test_chunk_boundary(self):
        """仅写必要输入列，第二日最早样本仍包含前日窗口。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = ["2026-07-11", "2026-07-12"]
            values = np.arange(2400) / 250
            table = pl.DataFrame({
                "symbol": ["BTCUSDT"] * 2400, "date": [dates[0]] * 1200 + [dates[1]] * 1200,
                "timestamp_ns": 1783814400000000000 - 120_000_000_000 + np.arange(2400, dtype=np.int64) * 100_000_000,
            })
            columns = ["imbalance_1", "imbalance_5", "imbalance_10", "trade_imbalance_100ms", "trade_imbalance_1s"]
            for index, column in enumerate(columns):
                write_parquet(table.with_columns(pl.Series(column, values * (index + 1))).lazy(), root / f"{column}.parquet",
                              {"kind": "feature", "dataset": {"dates": dates}, "definition": {}})
            config = {"min_fraction": 0.8, "tick_size": 0.1, "write_threads": 2, "output_interval_ns": 1_000_000_000, "aggregation_windows_s": [3, 5, 10, 20, 50, 100], "mean_diff_windows_s": [[3, 10], [5, 20], [5, 50], [10, 50], [10, 100], [20, 100]]}
            expressions = factor_module("gen001").factor_expressions(config)
            name = "book_obi_l1_mean_100s"
            compute_factor_chunk({
                "config": config, "features": str(root), "dates": [dates[1]], "batch": "gen001",
                "names": [name], "staging": str(root / "staging"), "index": 0,
                "metadata": {name: {"kind": "factor", "dataset": {"dates": dates}, "definition": {}}},
            })
            actual = pl.read_parquet(root / f"staging/0000/{name}.parquet")[name]
            expected = pl.DataFrame({"imbalance_1": values}).select(expressions[name].alias(name))[name].tail(1200).gather_every(10)
            np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
