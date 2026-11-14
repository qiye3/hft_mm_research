"""Gen041手算、缺失传播、跨午夜及未来扰动测试。"""

import unittest
import tempfile
from pathlib import Path
import json

import numpy as np
import polars as pl
import yaml
from polars.testing import assert_frame_equal

from mm_research.factors.gen041 import second_statistics, factor_expressions, window_statistic
from mm_research.data.storage import write_series_frames
from mm_research.models.gen041 import read_second_history, cache_model_inputs


class Gen041Test(unittest.TestCase):
    # 【非核心】新增模型输入必须严格滞后一秒，只拟合fit统计并保留缺失通道与旧列。
    def test_model_cache_lag_and_fit_only_scaling(self) -> None:
        """用可手算的十行数据核对E1/E2/E3拆分、上一秒读取和验证缺失处理。"""
        expressions, _ = factor_expressions()
        columns = list(expressions)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            fold = source / "e1/folds/final"
            fold.mkdir(parents=True)
            (source / "e0").mkdir()
            np.save(fold / "fit_rows.npy", [1, 2, 3])
            np.save(fold / "tune_rows.npy", [4])
            (source / "e0/splits.json").write_text(json.dumps({"segments": {
                "val_dev": {"start_row": 5, "stop_row": 7}, "val_select": {"start_row": 7, "stop_row": 9},
            }}))
            factors = root / "factors"
            factors.mkdir()
            for index, column in enumerate(columns):
                values = np.arange(10, dtype=float) + index
                if index == 0:
                    values[5] = np.nan
                pl.DataFrame({column: values}).write_parquet(factors / f"{column}.parquet")
            (root / "factors_ready.json").write_text(json.dumps({"columns": columns}))
            old = root / "old/all/inputs/180s"
            old.mkdir(parents=True)
            for split, count in [("fit", 3), ("tune", 1), ("val_dev", 2), ("val_select", 2)]:
                np.save(old / f"{split}_x.npy", np.full((count, 2), 77., dtype="float32"))
            config = {"source": str(source), "output": str(root / "output"), "factor_run": str(root),
                      "factor_folder": str(factors), "previous_macro_source": str(root / "old")}
            filename = root / "config.yaml"
            filename.write_text(yaml.safe_dump(config))
            cache_model_inputs(filename)
            for group, count in [("e1", 27), ("e2", 9), ("e3", 36)]:
                path = root / "output" / group / "all/inputs/180s"
                matrix = np.load(path / "val_dev_x.npy")
                self.assertEqual(matrix.shape, (2, 2 + count * 2))
                np.testing.assert_array_equal(matrix[:, :2], 77.)
            matrix = np.load(root / "output/e1/all/inputs/180s/val_dev_x.npy")
            self.assertAlmostEqual(matrix[0, 2], 3 / np.sqrt(2 / 3), places=5)
            self.assertEqual(matrix[1, 2], 0.)
            self.assertEqual(matrix[1, 2 + 27], 1.)

    # 【非核心】生产先筛整秒盘口的优化必须与直接100ms参考计算逐列一致。
    def test_production_seconds_match_reference(self) -> None:
        """使用真实单变量Parquet格式验证跨日读取和两类聚合的时间关联。"""
        raw = self.raw_sample()
        stored = raw.with_columns(
            pl.lit("BTCUSDT").alias("symbol"), pl.lit(.5).alias("imbalance_1"),
            pl.col("timestamp_ns").cast(pl.Datetime("ns")).dt.strftime("%Y-%m-%d").alias("date"),
        )
        columns = [column for column in stored.columns if column not in ["symbol", "date", "timestamp_ns"]]
        metadata = {column: {"dataset": {"interval_ns": 100_000_000}} for column in columns}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_series_frames([stored], folder, columns, metadata)
            optimized = read_second_history(folder, "1970-01-02")
        assert_frame_equal(optimized, second_statistics(raw))

    # 【非核心】构造跨午夜的完整100ms网格，盘口固定以便手算聚合与执行成本。
    def raw_sample(self) -> pl.DataFrame:
        """返回四秒与首个端点，主动买卖数量分别为每格2和1BTC。"""
        columns = {
            "timestamp_ns": np.arange(41, dtype=np.int64) * 100_000_000 + 86398_000_000_000,
            "mid_price": np.full(41, 60000.), "book_segment_id": np.ones(41),
            "buy_volume": np.full(41, 2.), "sell_volume": np.ones(41),
            "trade_small_buy_qty_100ms": np.full(41, .2),
            "trade_small_sell_qty_100ms": np.full(41, .1),
        }
        for side, sign, quantity in [("bid", -1, 3.), ("ask", 1, 1.)]:
            for level in range(1, 11):
                columns[f"{side}_price_{level}"] = np.full(41, 60000. + sign * level)
                columns[f"{side}_quantity_{level}"] = np.full(41, quantity)
        return pl.DataFrame(columns)

    # 【非核心】十个桶相加为一秒，盘口方向及买卖成本可独立手算。
    def test_seconds_and_missing_flow(self) -> None:
        """覆盖跨午夜不重置及单个未知成交桶导致整秒成交缺失。"""
        raw = self.raw_sample()
        seconds = second_statistics(raw)
        self.assertEqual(seconds["buy_volume"][1], 20.)
        self.assertEqual(seconds["sell_volume"][1], 10.)
        self.assertEqual(seconds["obi"][1], .5)
        self.assertEqual(seconds["bid_delta"][2], 0.)
        self.assertAlmostEqual(seconds["cost_gap"][1], 0., places=9)
        changed = raw.with_row_index().with_columns(
            pl.when(pl.col("index") == 5).then(None).otherwise(pl.col("buy_volume")).alias("buy_volume"),
        )
        self.assertIsNone(second_statistics(changed)["buy_volume"][1])
        thin = raw.with_columns(pl.lit(.001).alias("ask_quantity_1"))
        thin = thin.with_columns([pl.lit(.001).alias(f"ask_quantity_{level}") for level in range(2, 11)])
        self.assertIsNone(second_statistics(thin)["cost_gap"][1])
        interrupted = raw.with_row_index().with_columns(
            pl.when(pl.col("index") >= 10).then(2.).otherwise(1.).alias("book_segment_id"),
        )
        self.assertIsNone(second_statistics(interrupted)["bid_delta"][1])
        self.assertEqual(second_statistics(interrupted)["bid_delta"][2], 0.)

    # 【非核心】高成交量秒必须获得相应权重，已知无活动与未知缺失不能混淆。
    def test_weighted_ratio_and_zero_activity(self) -> None:
        """独立计算累计量比值，并检查全零与全空两个不同结果。"""
        expression, inactive = window_statistic("buy_volume", "quantity", 2, 1.)
        frame = pl.DataFrame({"buy_volume": [9., 0., 0., None, None], "sell_volume": [0., 1., 0., None, None]})
        result = frame.select(expression.alias("value"), inactive.alias("inactive"))
        self.assertAlmostEqual(result["value"][1], .8)
        self.assertIsNone(result["value"][-1])
        zero = pl.DataFrame({"buy_volume": [0., 0.], "sell_volume": [0., 0.]})
        self.assertEqual(zero.select(expression).item(-1, 0), 0.)
        self.assertTrue(zero.select(inactive).item(-1, 0))

    # 【非核心】全部36列必须保持因果前缀不变，分位数直接对有效整秒观测计算。
    def test_causal_prefix_and_quantile(self) -> None:
        """扰动未来全部输入不改变过去，验证长分位数的独立NumPy参考。"""
        count = 4000
        values = np.arange(count, dtype=float) + 1.
        frame = pl.DataFrame({
            "obi": values / count, "net_delta": np.sin(values), "bid_delta": np.cos(values),
            "ask_delta": -np.cos(values), "buy_volume": values, "sell_volume": values / 2,
            "trade_small_buy_qty_100ms": values / 10, "trade_small_sell_qty_100ms": values / 20,
            "agreement": np.ones(count), "cost_gap": values / count, "notional_depth": values,
        })
        expressions, _ = factor_expressions()
        self.assertEqual(len(expressions), 36)
        original = frame.select([value.alias(name) for name, value in expressions.items()])
        changed = pl.concat([frame.head(2000), frame.tail(2000) * -2])
        modified = changed.select([value.alias(name) for name, value in expressions.items()])
        assert_frame_equal(original.head(2000), modified.head(2000))
        self.assertAlmostEqual(original["book_depth_notional_p10_3600s"][-1], np.quantile(values[-3600:], .1))


if __name__ == "__main__":
    unittest.main()
