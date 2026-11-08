"""核对有效样本、平均排名相关系数、重复值分组及报告入口。"""

from pathlib import Path
import datetime as dt
import tempfile
import unittest

import numpy as np
import polars as pl
import yaml

from mm_research.data.storage import write_parquet
from mm_research.evaluation.metrics import correlation_statistics, distribution_statistics, grouped_returns
from mm_research.evaluation.report import fit_group_boundaries, run_evaluation


class EvaluationTests(unittest.TestCase):
    """使用人工收益核对指标，再通过真实文件入口生成报告。"""

    # 【非核心】区分因子与标签缺失，相关系数仅使用共同有效样本。
    def test_correlations_missing_and_constant(self):
        """线性样本相关为1，常数样本产生明确的不可计算原因。"""
        table = pl.DataFrame({"value": [0.0, 1.0, 1.0, 2.0, None], "target": [0.0, 2.0, 2.0, 4.0, 9.0],
                              "label_valid": [True, True, True, True, True]})
        metrics = correlation_statistics(table)
        self.assertEqual(metrics["paired_rows"], 4)
        self.assertAlmostEqual(metrics["ic"], 1)
        self.assertAlmostEqual(metrics["rank_ic"], 1)
        uneven = table.with_columns(pl.Series("target", [4.0, 1.0, 3.0, 2.0, 9.0]))
        expected_rank_ic = np.corrcoef([1.0, 2.5, 2.5, 4.0], [4.0, 1.0, 3.0, 2.0])[0, 1]
        self.assertAlmostEqual(correlation_statistics(uneven)["rank_ic"], expected_rank_ic)
        self.assertEqual(distribution_statistics(table)["zero_rate"], 0.25)
        constant = table.with_columns(pl.lit(1.0).alias("value"))
        self.assertEqual(correlation_statistics(constant)["reason"], "constant_factor")
        self.assertIsNone(correlation_statistics(constant)["ic"])

    # 【非核心】大量重复值保持共同归组，恒定因子只形成一组。
    def test_tied_groups(self):
        """分组样本数守恒，只有两个唯一值时实际组数小于五。"""
        table = pl.DataFrame({"value": [0.0] * 80 + [1.0] * 20, "target": [0.0] * 80 + [2.0] * 20,
                              "label_valid": [True] * 100})
        groups = grouped_returns(table, 5, {"mode": "sign", "cutoffs": []})
        self.assertEqual(groups["samples"].sum(), 100)
        self.assertEqual(groups.height, 2)
        self.assertEqual(groups["factor_mean"].to_list(), [0.0, 1.0])
        self.assertEqual(groups["mean_return_bps"].to_list(), [0.0, 2.0])

    # 【非核心】分组阈值也剔除训练末尾300秒，隔离段极端值不进入冻结阈值。
    def test_group_thresholds_respect_training_purge(self):
        """只用隔离区间以前的因子拟定分位数，验证期也不参与。"""
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "pressure.parquet"
            boundary = int(dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
            pl.DataFrame({
                "date": ["2026-05-31"] * 5 + ["2026-06-01"],
                "timestamp_ns": [boundary - seconds * 1_000_000_000 for seconds in [304, 303, 302, 301, 300, -1]],
                "pressure": [1.0, 2.0, 3.0, 4.0, 10000.0, 20000.0],
            }).write_parquet(filename)
            result = fit_group_boundaries(filename, {
                "dates": ["2026-05-31", "2026-06-01"], "train_end": "2026-05-31",
                "purge_ns": 300_000_000_000, "groups": 5,
            })
            self.assertEqual(result["min"], 1.0)
            self.assertEqual(result["max"], 4.0)
            np.testing.assert_allclose(result["cutoffs"], [1.6, 2.2, 2.8, 3.4])

    # 【非核心】通过真实Parquet和配置入口生成表格与图，核对收益单位。
    def test_report_entrypoint(self):
        """输入已知因子及两倍bps收益，报告应有正相关和正确图表。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "test_factor"
            date = "2026-07-12"
            dataset = {"dates": [date], "symbol": "BTCUSDT"}
            keys = {"symbol": ["BTCUSDT"] * 100, "date": [date] * 100,
                    "timestamp_ns": np.arange(100, dtype=np.int64) * 1_000_000_000}
            factor = pl.DataFrame({**keys, name: np.arange(100, dtype=float)})
            write_parquet(factor.lazy(), root / "factors/test/test_factor.parquet",
                          {"kind": "factor", "dataset": dataset, "definition": {"dates": [date]}})
            labels = pl.DataFrame({**keys, "label_valid": [True] * 100, "label_end_timestamp_ns": np.arange(100, dtype=np.int64) * 1_000_000_000 + 1_000_000_000, "y_mid_1s_bps": np.arange(100) * 2.0})
            write_parquet(labels.lazy(), root / f"labels/test/horizon=1s/{date}.parquet",
                          {"kind": "labels", "dataset": dataset, "horizon_ns": 1_000_000_000})
            long_labels = labels.rename({"y_mid_1s_bps": "y_mid_10s_bps"}).with_columns(
                (pl.col("timestamp_ns") + 10_000_000_000).alias("label_end_timestamp_ns"),
            )
            write_parquet(long_labels.lazy(), root / f"labels/test/horizon=10s/{date}.parquet",
                          {"kind": "labels", "dataset": dataset, "horizon_ns": 10_000_000_000})
            config = {"stage": "evaluation", "factor_root": str(root / "factors"), "factor_version": "test",
                      "label_root": str(root / "labels"), "label_version": "test", "dates": [date],
                      "factors": [name], "horizons_ns": [1_000_000_000, 10_000_000_000], "groups": 5, "output": str(root / "results"),
                      "train_end": date, "validation_end": date, "workers": 1, "purge_ns": 300_000_000_000,
                      "output_interval_ns": 1_000_000_000, "factor_batch_size": 6}
            filename = root / "evaluation.yaml"
            filename.write_text(yaml.safe_dump(config))
            result = run_evaluation(filename)
            self.assertAlmostEqual(result["summary"].filter(pl.col("sampling") == "all")["ic_mean"][0], 1)
            self.assertAlmostEqual(result["summary"].filter(pl.col("sampling") == "all")["top_bottom_bps"][0], 160)
            group_results = pl.read_parquet(root / "results/test_factor/group_returns.parquet").filter(pl.col("horizon_ns") == 10_000_000_000)
            self.assertEqual(group_results.filter(pl.col("sampling") == "all")["samples"].sum(), 100)
            self.assertEqual(group_results.filter(pl.col("sampling") == "nonoverlap")["samples"].sum(), 10)
            self.assertTrue((root / "results/report.html").exists())
            self.assertTrue((root / "results/test_factor/diagnostics.png").exists())


if __name__ == "__main__":
    unittest.main()
