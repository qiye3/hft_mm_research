"""用独立数组参考核对Gen001公式、缺失阈值、跨午夜与前缀因果性。"""

import unittest

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from mm_research.factors.gen001 import INPUT_COLUMNS, factor_expressions


class FactorTests(unittest.TestCase):
    """六个压力序列使用不同输入，但共用原生窗口定义。"""

    # 【非核心】逐行独立计算六个均值差，检查空值、NaN与午夜前后结果。
    def test_mean_differences_missing_values_and_midnight(self):
        """输入带缺失值的连续120点，结果与NumPy逐窗参考完全一致。"""
        config = {"short_window": 10, "long_window": 50, "short_min_samples": 8, "long_min_samples": 40}
        numbers = np.sin(np.arange(120) / 9)
        numbers[[17, 40, 55]] = np.nan
        values = {name: numbers * (index + 1) / 5 for index, name in enumerate(INPUT_COLUMNS)}
        values["timestamp_ns"] = 86_400_000_000_000 - 6_000_000_000 + np.arange(120) * 100_000_000
        table = pl.DataFrame(values)
        expressions = factor_expressions(config)
        result = table.select([expression.alias(name) for name, expression in expressions.items()])
        sources = [values[name] for name in INPUT_COLUMNS]
        sources.append(values["trade_imbalance_1s"] - values["imbalance_5"])
        for name, source in zip(expressions, sources):
            expected = []
            for index in range(len(source)):
                short = source[max(0, index - 9):index + 1]
                long = source[max(0, index - 49):index + 1]
                valid = np.isfinite(short).sum() >= 8 and np.isfinite(long).sum() >= 40
                expected.append(np.nanmean(short) - np.nanmean(long) if valid else np.nan)
            np.testing.assert_allclose(result[name].to_numpy(), expected, atol=1e-12, rtol=0, equal_nan=True)
            self.assertIsNotNone(result[name][60])
        prefix = table.head(75).select([expression.alias(name) for name, expression in expressions.items()])
        assert_frame_equal(prefix, result.head(75), check_exact=True)

    # 【非核心】非法窗口组合在构造公式前明确失败。
    def test_invalid_windows(self):
        """最少有效样本数超过窗口时拒绝配置。"""
        with self.assertRaises(ValueError):
            factor_expressions({"short_window": 10, "long_window": 50, "short_min_samples": 11, "long_min_samples": 40})


if __name__ == "__main__":
    unittest.main()
