"""核对有效样本、平均排名相关系数、重复值分组及报告入口。"""

from pathlib import Path
import datetime as dt
import json
import tempfile
import unittest

import numpy as np
import polars as pl
import yaml

from mm_research.data.storage import write_parquet
from mm_research.evaluation.metrics import (
    bootstrap_interval, bootstrap_summary, correlation_statistics, distribution_statistics,
    extreme_tail_days, grouped_returns, group_quality_statistics, oriented_statistics,
    single_factor_ols, train_ic_direction,
)
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
             "factor_lag_ns": 0})
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
                    "timestamp_ns": int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000 + (np.arange(100, dtype=np.int64) * 1_000_000_000)}
            factor = pl.DataFrame({**keys, name: np.arange(100, dtype=float)})
            write_parquet(factor.lazy(), root / "factors/test/test_factor.parquet",
                          {"kind": "factor", "dataset": dataset, "definition": {"dates": [date]}})
            labels = pl.DataFrame({**keys, "label_valid": [True] * 100, "label_end_timestamp_ns": int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000 + (np.arange(100, dtype=np.int64) * 1_000_000_000 + 1_000_000_000), "y_mid_1s_bps": np.arange(100) * 2.0})
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
                      "output_interval_ns": 1_000_000_000, "factor_batch_size": 6, "factor_lag_ns": 0, "validation_dev_end": date}
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


class DirectionMetricsTests(unittest.TestCase):
    """核对训练方向、方向一致率、sign组价差和方向化价差的汇总口径。"""

    DAILY_COLUMNS = ["factor", "date", "split", "horizon_ns", "sampling", "ic", "rank_ic"]
    RETURN_COLUMNS = ["factor", "date", "split", "horizon_ns", "sampling", "group", "mean_return_bps"]

    # 【非核心】无分组收益时sign价差保持null，各测试共用空表结构。
    def no_returns(self) -> pl.DataFrame:
        """返回带完整列类型的空分组收益表。"""
        return pl.DataFrame(schema={"factor": pl.String, "date": pl.String, "split": pl.String,
                                    "horizon_ns": pl.Int64, "sampling": pl.String, "group": pl.Int32,
                                    "mean_return_bps": pl.Float64})

    # 【非核心】连续负IC训练方向为-1，正IC日比例为0但方向一致率为1。
    def test_negative_ic_direction(self):
        """训练期全负IC确定-1方向，测试期同号日期全部计为一致。"""
        daily = pl.DataFrame({
            "factor": ["f"] * 4, "date": [f"2026-01-0{index}" for index in range(1, 5)],
            "split": ["train", "train", "test", "test"],
            "horizon_ns": [1_000_000_000] * 4, "sampling": ["all"] * 4,
            "ic": [-0.1, -0.2, -0.3, -0.4], "rank_ic": [-0.1, -0.2, -0.3, -0.4],
        }, schema_overrides={"ic": pl.Float64, "rank_ic": pl.Float64})
        rows = []
        for day, split, low, high in [("2026-01-01", "train", 1.0, -1.0), ("2026-01-02", "train", 1.0, -1.0),
                                      ("2026-01-03", "test", 2.0, -2.0), ("2026-01-04", "test", 2.0, -2.0)]:
            rows.append({"factor": "f", "date": day, "split": split, "horizon_ns": 1_000_000_000,
                         "sampling": "all", "group": 1, "mean_return_bps": low})
            rows.append({"factor": "f", "date": day, "split": split, "horizon_ns": 1_000_000_000,
                         "sampling": "all", "group": 3, "mean_return_bps": high})
        returns = pl.DataFrame(rows, schema_overrides={"mean_return_bps": pl.Float64})
        directions = train_ic_direction(daily)
        self.assertEqual(directions["train_ic_direction"].to_list(), [-1])
        oriented = oriented_statistics(daily, returns, directions)
        self.assertEqual(oriented["train_ic_direction"].unique().to_list(), [-1])
        # 训练期和测试期都与-1方向同号，一致率均为1。
        self.assertTrue((oriented["direction_consistency_ratio"] == 1.0).all())
        # 正值组收益减负值组：训练日-2.0，测试日-4.0，各自等权。
        self.assertAlmostEqual(oriented.filter(pl.col("split") == "train")["sign_group_spread_bps"][0], -2.0)
        self.assertAlmostEqual(oriented.filter(pl.col("split") == "test")["sign_group_spread_bps"][0], -4.0)

    # 【非核心】验证期和测试期不得用自己的IC重新确定方向。
    def test_validation_keeps_train_direction(self):
        """训练方向-1而验证期IC全为正时，方向仍是-1且一致率为0。"""
        daily = pl.DataFrame({
            "factor": ["f"] * 3, "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "split": ["train", "validation", "validation"],
            "horizon_ns": [1_000_000_000] * 3, "sampling": ["all"] * 3,
            "ic": [-0.5, 0.2, 0.3], "rank_ic": [-0.5, 0.2, 0.3],
        }, schema_overrides={"ic": pl.Float64, "rank_ic": pl.Float64})
        directions = train_ic_direction(daily)
        self.assertEqual(directions["train_ic_direction"].to_list(), [-1])
        oriented = oriented_statistics(daily, self.no_returns(), directions)
        validation = oriented.filter(pl.col("split") == "validation")
        self.assertEqual(validation["train_ic_direction"][0], -1)
        self.assertAlmostEqual(validation["direction_consistency_ratio"][0], 0.0)
        self.assertIsNone(validation["sign_group_spread_bps"][0])

    # 【非核心】零IC计入有效日但不算方向一致；IC为0的训练均值方向为null。
    def test_zero_ic_and_null_direction(self):
        """方向null时一致率与方向化价差为null，零IC日拉低一致率。"""
        daily = pl.DataFrame({
            "factor": ["f"] * 4, "date": [f"2026-01-0{index}" for index in range(1, 5)],
            "split": ["train", "train", "test", "test"],
            "horizon_ns": [1_000_000_000] * 4, "sampling": ["all"] * 4,
            "ic": [0.2, -0.2, 0.5, 0.0], "rank_ic": [0.1, -0.1, 0.5, 0.0],
        }, schema_overrides={"ic": pl.Float64, "rank_ic": pl.Float64})
        directions = train_ic_direction(daily)
        self.assertIsNone(directions["train_ic_direction"][0])
        oriented = oriented_statistics(daily, self.no_returns(), directions)
        self.assertIsNone(oriented["direction_consistency_ratio"][0])
        # 训练均值恰好为正时，测试日零IC计入分母不计分子。
        positive = daily.with_columns(pl.Series("ic", [0.3, 0.1, 0.5, 0.0]))
        oriented_positive = oriented_statistics(positive, self.no_returns(), train_ic_direction(positive))
        # 测试日IC为0.5和0.0：零IC计入分母不计分子，一致率0.5。
        self.assertAlmostEqual(oriented_positive.filter(pl.col("split") == "test")["direction_consistency_ratio"][0], 0.5)

    # 【非核心】缺少任一端点组的日期不参与sign组价差。
    def test_sign_spread_missing_endpoint(self):
        """三天中一天缺正值组，sign组价差只用其余两天等权平均。"""
        rows = []
        for day, groups in [("2026-01-01", [(1, 1.0), (3, 3.0)]), ("2026-01-02", [(1, 2.0), (3, 6.0)]),
                            ("2026-01-03", [(2, 0.0)])]:
            for group, bps in groups:
                rows.append({"factor": "f", "date": day, "split": "train", "horizon_ns": 1_000_000_000,
                             "sampling": "all", "group": group, "mean_return_bps": float(bps)})
        returns = pl.DataFrame(rows, schema_overrides={"mean_return_bps": pl.Float64})
        daily = pl.DataFrame({
            "factor": ["f"] * 3, "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "split": ["train", "train", "train"], "horizon_ns": [1_000_000_000] * 3, "sampling": ["all"] * 3,
            "ic": [0.2, 0.3, 0.4], "rank_ic": [0.2, 0.3, 0.4],
        }, schema_overrides={"ic": pl.Float64, "rank_ic": pl.Float64})
        oriented = oriented_statistics(daily, returns, train_ic_direction(daily))
        self.assertAlmostEqual(oriented["sign_group_spread_bps"][0], 3.0)
        self.assertEqual(oriented["sign_spread_days"][0], 2)

    # 【非核心】五分组旧top_bottom_bps不变，方向化价差=原值×训练方向。
    def test_quantile_summary_unchanged_and_oriented(self):
        """正常五分组走完整评估入口，旧指标保持原数，新列正确补充。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "test_factor"
            date = "2026-07-12"
            dataset = {"dates": [date], "symbol": "BTCUSDT"}
            keys = {"symbol": ["BTCUSDT"] * 100, "date": [date] * 100,
                    "timestamp_ns": int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000 + (np.arange(100, dtype=np.int64) * 1_000_000_000)}
            factor = pl.DataFrame({**keys, name: np.arange(100, dtype=float)})
            write_parquet(factor.lazy(), root / "factors/test/test_factor.parquet",
                          {"kind": "factor", "dataset": dataset, "definition": {"dates": [date]}})
            labels = pl.DataFrame({**keys, "label_valid": [True] * 100,
                                   "label_end_timestamp_ns": int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000 + (np.arange(100, dtype=np.int64) * 1_000_000_000 + 1_000_000_000),
                                   "y_mid_1s_bps": np.arange(100) * 2.0})
            write_parquet(labels.lazy(), root / "labels/test/horizon=1s/{date}.parquet".replace("{date}", date),
                          {"kind": "labels", "dataset": dataset, "horizon_ns": 1_000_000_000})
            config = {"stage": "evaluation", "factor_root": str(root / "factors"), "factor_version": "test",
                      "label_root": str(root / "labels"), "label_version": "test", "dates": [date],
                      "factors": [name], "horizons_ns": [1_000_000_000], "groups": 5, "output": str(root / "results"),
                      "train_end": date, "validation_end": date, "workers": 1, "purge_ns": 300_000_000_000,
                      "output_interval_ns": 1_000_000_000, "factor_batch_size": 6, "factor_lag_ns": 0, "validation_dev_end": date}
            filename = root / "evaluation.yaml"
            filename.write_text(yaml.safe_dump(config))
            run_evaluation(filename)
            summary = pl.read_parquet(root / "results/test_factor/summary.parquet").filter(pl.col("sampling") == "all")
            self.assertAlmostEqual(summary["top_bottom_bps"][0], 160)
            self.assertAlmostEqual(summary["ic_mean"][0], 1)
            self.assertEqual(summary["train_ic_direction"][0], 1)
            self.assertAlmostEqual(summary["direction_consistency_ratio"][0], 1.0)
            self.assertAlmostEqual(summary["oriented_group_spread_bps"][0], 160)
            self.assertIsNone(summary["sign_group_spread_bps"][0])
            self.assertIsNone(summary["sign_spread_days"][0])
            self.assertEqual(json.loads((root / "results/test_factor/completed.json").read_text())["definition_version"],
                             "coarse_windows_log_bps_seven_horizons_v3")

    # 【非核心】重复分位阈值保持主五分组定义，不足五组的指标明确为空。
    def test_sign_mode_evaluation_entrypoint(self):
        """负零正重复值保持quantile口径，真实入口不自动替换为其他分组。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "tied_factor"
            dates = ["2026-07-12", "2026-07-13"]
            factor_frames = []
            for offset, date in enumerate(dates):
                keys = {"symbol": ["BTCUSDT"] * 100, "date": [date] * 100,
                        "timestamp_ns": int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000 + ((np.arange(100, dtype=np.int64) + offset * 100) * 1_000_000_000)}
                values = np.full(100, -1.0)
                values[20:80] = 0.0
                values[80:] = 1.0
                factor_frames.append(pl.DataFrame({**keys, name: values}))
                labels = pl.DataFrame({**keys, "label_valid": [True] * 100,
                                       "label_end_timestamp_ns": int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000 + ((np.arange(100, dtype=np.int64) + offset * 100)
                                       * 1_000_000_000 + 1_000_000_000),
                                       "y_mid_1s_bps": values * (2.0 if offset == 0 else 4.0)})
                write_parquet(labels.lazy(), root / f"labels/test/horizon=1s/{date}.parquet",
                              {"kind": "labels", "dataset": {"dates": dates, "symbol": "BTCUSDT"},
                               "horizon_ns": 1_000_000_000})
            dataset = {"dates": dates, "symbol": "BTCUSDT"}
            write_parquet(pl.concat(factor_frames).lazy(), root / f"factors/test/{name}.parquet",
                          {"kind": "factor", "dataset": dataset, "definition": {"dates": dates}})
            config = {"stage": "evaluation", "factor_root": str(root / "factors"), "factor_version": "test",
                      "label_root": str(root / "labels"), "label_version": "test", "dates": dates,
                      "factors": [name], "horizons_ns": [1_000_000_000], "groups": 5, "output": str(root / "results"),
                      "train_end": dates[0], "validation_end": dates[0], "workers": 1, "purge_ns": 300_000_000_000,
                      "output_interval_ns": 1_000_000_000, "factor_batch_size": 6, "factor_lag_ns": 0, "validation_dev_end": dates[0]}
            filename = root / "evaluation.yaml"
            filename.write_text(yaml.safe_dump(config))
            run_evaluation(filename)
            self.assertEqual(json.loads((root / f"results/{name}/group_boundaries.json").read_text())["mode"], "quantile")
            summary = pl.read_parquet(root / f"results/{name}/summary.parquet").filter(
                (pl.col("split") == "test") & (pl.col("sampling") == "all"))
            # 验证日标签为±4bps，正组4、负组-4，sign价差8；训练方向为正。
            self.assertIsNone(summary["sign_group_spread_bps"][0])
            self.assertIsNone(summary["sign_spread_days"][0])
            self.assertEqual(summary["train_ic_direction"][0], 1)
            self.assertIsNone(summary["oriented_group_spread_bps"][0])
            self.assertAlmostEqual(summary["direction_consistency_ratio"][0], 1.0)
            # 旧top_bottom口径保持原始定义：sign模式下完整三组日仍为G3-G1，不与新指标混算。
            self.assertIsNone(summary["top_bottom_bps"][0])


class BootstrapIntervalTests(unittest.TestCase):
    """核对交易日重采样区间的确定性、方向性、缺失处理和切分隔离。"""

    # 【非核心】辅助构造逐日IC与端点分组收益的最小表。
    @staticmethod
    def frames(ic_by_split, spread_rows=None):
        """ic_by_split形如{'train':[...]}, 日期按2026-01-01起顺序编号。"""
        daily_rows, returns_rows = [], []
        for split, values in ic_by_split.items():
            for index, value in enumerate(values):
                daily_rows.append({"factor": "f", "date": f"2026-01-{index + 1:02d}", "split": split,
                                   "horizon_ns": 1_000_000_000, "sampling": "all",
                                   "ic": value, "rank_ic": value})
        for row in spread_rows or []:
            returns_rows.append({**row, "factor": "f", "horizon_ns": 1_000_000_000, "sampling": "all"})
        daily = pl.DataFrame(daily_rows, schema_overrides={"ic": pl.Float64, "rank_ic": pl.Float64})
        returns = pl.DataFrame(returns_rows, schema_overrides={"mean_return_bps": pl.Float64},
                               schema={"factor": pl.String, "date": pl.String, "split": pl.String,
                                       "horizon_ns": pl.Int64, "sampling": pl.String, "group": pl.Int32,
                                       "mean_return_bps": pl.Float64}) if returns_rows else pl.DataFrame(
            schema={"factor": pl.String, "date": pl.String, "split": pl.String, "horizon_ns": pl.Int64,
                    "sampling": pl.String, "group": pl.Int32, "mean_return_bps": pl.Float64})
        return daily, returns

    # 【非核心】固定种子下重复运行，区间逐位一致。
    def test_deterministic_seed(self):
        """同一输入两次调用bootstrap_summary输出完全相同。"""
        daily, returns = self.frames({"train": [0.1, 0.2, 0.3, -0.1, 0.05]})
        first = bootstrap_summary(daily, returns, 5)
        second = bootstrap_summary(daily, returns, 5)
        self.assertTrue(first.equals(second))

    # 【非核心】正、负、零相关序列的区间方向正确。
    def test_positive_negative_zero(self):
        """全正IC区间下界大于0，全负上界小于0，零均值对称序列区间跨0。"""
        positive = bootstrap_interval([0.2, 0.3, 0.25, 0.28, 0.22, 0.27])
        self.assertGreater(positive["ci95_low"], 0)
        negative = bootstrap_interval([-0.2, -0.3, -0.25, -0.28, -0.22, -0.27])
        self.assertLess(negative["ci95_high"], 0)
        zero = bootstrap_interval([0.5, -0.5, 0.4, -0.4, 0.45, -0.45])
        self.assertLessEqual(zero["ci95_low"], 0)
        self.assertGreaterEqual(zero["ci95_high"], 0)
        self.assertEqual(positive["valid_days"], 6)

    # 【非核心】缺失日、常数因子和有效日不足时输出null而非0。
    def test_missing_and_insufficient(self):
        """null日剔除后不足两日返回null区间，常数因子的IC日全部缺失。"""
        single = bootstrap_interval([0.3])
        self.assertIsNone(single["ci95_low"])
        self.assertIsNone(single["ci95_high"])
        self.assertEqual(single["valid_days"], 1)
        empty = bootstrap_interval([None, None, None])
        self.assertIsNone(empty["ci95_low"])
        self.assertEqual(empty["valid_days"], 0)
        daily, returns = self.frames({"train": [None, None, 0.1], "test": [0.2]})
        summary = bootstrap_summary(daily, returns, 5).sort("split")
        train_row = summary.filter(pl.col("split") == "train")
        self.assertIsNone(train_row["group_spread_ci95_low"][0])
        self.assertEqual(train_row["bootstrap_valid_days"][0], 1)
        self.assertIsNone(summary.filter(pl.col("split") == "test")["ic_ci95_low"][0])

    # 【非核心】端点组差先同日配对再重采样，且各切分不混用。
    def test_spread_pairs_same_day_and_split_isolation(self):
        """每日组差恒为2时区间退化为[2,2]；两个split的IC区间围绕各自均值。"""
        spreads = []
        for day in range(1, 5):
            spreads.append({"date": f"2026-01-{day:02d}", "split": "train", "group": 1, "mean_return_bps": 1.0 + day})
            spreads.append({"date": f"2026-01-{day:02d}", "split": "train", "group": 5, "mean_return_bps": 3.0 + day})
        daily, returns = self.frames({"train": [0.5] * 4, "test": [0.01] * 4}, spreads)
        summary = bootstrap_summary(daily, returns, 5)
        train = summary.filter(pl.col("split") == "train")
        # 每日差恒为2，任何重采样均值都是2。
        self.assertAlmostEqual(train["group_spread_ci95_low"][0], 2.0)
        self.assertAlmostEqual(train["group_spread_ci95_high"][0], 2.0)
        # test没有分组行，组差区间为null，IC区间围绕0.01且远离train的0.5。
        test = summary.filter(pl.col("split") == "test")
        self.assertIsNone(test["group_spread_ci95_low"][0])
        self.assertGreater(test["ic_ci95_low"][0], 0.005)
        self.assertLess(test["ic_ci95_high"][0], 0.4)

    # 【非核心】通过完整入口核对区间与现有summary口径一致，旧指标不变。
    def test_summary_consistency_via_entrypoint(self):
        """多日正相关因子走真实评估，区间包含对应均值且旧列保持不变。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "test_factor"
            dates = ["2026-07-12", "2026-07-13", "2026-07-14", "2026-07-15"]
            frames = []
            for offset, date in enumerate(dates):
                keys = {"symbol": ["BTCUSDT"] * 100, "date": [date] * 100,
                        "timestamp_ns": int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000 + ((np.arange(100, dtype=np.int64) + offset * 100) * 1_000_000_000)}
                factor = np.arange(100, dtype=float)
                frames.append(pl.DataFrame({**keys, name: factor}))
                labels = pl.DataFrame({**keys, "label_valid": [True] * 100,
                                       "label_end_timestamp_ns": int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000 + ((np.arange(100, dtype=np.int64) + offset * 100)
                                       * 1_000_000_000 + 1_000_000_000),
                                       "y_mid_1s_bps": factor * 2.0})
                write_parquet(labels.lazy(), root / f"labels/test/horizon=1s/{date}.parquet",
                              {"kind": "labels", "dataset": {"dates": dates, "symbol": "BTCUSDT"},
                               "horizon_ns": 1_000_000_000})
            write_parquet(pl.concat(frames).lazy(), root / f"factors/test/{name}.parquet",
                          {"kind": "factor", "dataset": {"dates": dates, "symbol": "BTCUSDT"},
                           "definition": {"dates": dates}})
            config = {"stage": "evaluation", "factor_root": str(root / "factors"), "factor_version": "test",
                      "label_root": str(root / "labels"), "label_version": "test", "dates": dates,
                      "factors": [name], "horizons_ns": [1_000_000_000], "groups": 5, "output": str(root / "results"),
                      "train_end": dates[1], "validation_end": dates[2], "workers": 1, "purge_ns": 300_000_000_000,
                      "output_interval_ns": 1_000_000_000, "factor_batch_size": 6, "factor_lag_ns": 0, "validation_dev_end": dates[2]}
            (root / "evaluation.yaml").write_text(yaml.safe_dump(config))
            run_evaluation(root / "evaluation.yaml")
            summary = pl.read_parquet(root / f"results/{name}/summary.parquet").filter(
                (pl.col("split") == "test") & (pl.col("sampling") == "all"))
            # 旧口径不变：完美正相关IC为1，五组完整日高低差160。
            self.assertAlmostEqual(summary["ic_mean"][0], 1)
            self.assertAlmostEqual(summary["top_bottom_bps"][0], 160)
            # 测试期只有一个交易日，区间按定义返回null且日数为1，不用0冒充。
            self.assertEqual(summary["bootstrap_valid_days"][0], 1)
            self.assertIsNone(summary["ic_ci95_low"][0])
            self.assertIsNone(summary["group_spread_ci95_low"][0])
            # 训练期两天：重采样日数等于有效IC日数，区间覆盖点估计。
            train = pl.read_parquet(root / f"results/{name}/summary.parquet").filter(
                (pl.col("split") == "train") & (pl.col("sampling") == "all"))
            self.assertEqual(train["bootstrap_valid_days"][0], train["valid_ic_days"][0])
            self.assertLessEqual(train["ic_ci95_low"][0], train["ic_mean"][0])
            self.assertGreaterEqual(train["ic_ci95_high"][0], train["ic_mean"][0])
            self.assertLessEqual(train["group_spread_ci95_low"][0], train["top_bottom_bps"][0])
            self.assertGreaterEqual(train["group_spread_ci95_high"][0], train["top_bottom_bps"][0])
            self.assertEqual(json.loads((root / f"results/{name}/completed.json").read_text())["definition_version"],
                             "coarse_windows_log_bps_seven_horizons_v3")
            # 各split日数互不混用：训练2天、验证与测试各1天。
            all_rows = pl.read_parquet(root / f"results/{name}/summary.parquet").filter(pl.col("sampling") == "all")
            self.assertEqual(sorted(all_rows["bootstrap_valid_days"].to_list()), [1, 1, 2])


class GroupQualityAndOlsTests(unittest.TestCase):
    """核对分组单调性、极端分位差与单因子OLS的时间分离和退化处理。"""

    # 【非核心】构造若干天五组（或三组）逐日分组收益。
    @staticmethod
    def returns_frame(daily_groups, split="train"):
        """daily_groups形如{'2026-01-01': {1: 1.0, ..., 5: 5.0}}。"""
        rows = []
        for date, groups in daily_groups.items():
            for group, bps in groups.items():
                rows.append({"factor": "f", "date": date, "split": split, "horizon_ns": 1_000_000_000,
                             "sampling": "all", "group": group, "mean_return_bps": float(bps)})
        return pl.DataFrame(rows, schema_overrides={"mean_return_bps": pl.Float64})

    @staticmethod
    def directions(value):
        """返回单个方向行，供纯函数直接调用。"""
        return pl.DataFrame({"horizon_ns": [1_000_000_000], "sampling": ["all"],
                             "train_ic_direction": [value]}, schema_overrides={"train_ic_direction": pl.Int32})

    @staticmethod
    def daily_ic(dates, split, value):
        """返回逐日IC表。"""
        return pl.DataFrame({"factor": ["f"] * len(dates), "date": dates, "split": split,
                             "horizon_ns": [1_000_000_000] * len(dates), "sampling": ["all"] * len(dates),
                             "ic": [float(value)] * len(dates), "rank_ic": [float(value)] * len(dates)},
                            schema_overrides={"ic": pl.Float64, "rank_ic": pl.Float64})

    # 【非核心】递增、递减、非单调五组的Spearman单调性方向正确。
    def test_monotonicity_directions(self):
        """严格递增为+1，严格递减为-1，中间折返的序列小于1。"""
        increasing = self.returns_frame({"2026-01-01": {g: float(g) for g in range(1, 6)},
                                         "2026-01-02": {g: float(g) + 1 for g in range(1, 6)}})
        result = group_quality_statistics(increasing, self.directions(1), 5)
        self.assertAlmostEqual(result["group_monotonicity"][0], 1.0)
        self.assertAlmostEqual(result["monotonic_days_ratio"][0], 1.0)
        self.assertAlmostEqual(result["group_spread_positive_days"][0], 1.0)
        decreasing = self.returns_frame({"2026-01-01": {g: float(6 - g) for g in range(1, 6)},
                                         "2026-01-02": {g: float(5 - g) for g in range(1, 6)}})
        result = group_quality_statistics(decreasing, self.directions(-1), 5)
        self.assertAlmostEqual(result["group_monotonicity"][0], -1.0)
        self.assertAlmostEqual(result["monotonic_days_ratio"][0], 1.0)
        self.assertAlmostEqual(result["group_spread_positive_days"][0], 1.0)
        # 组3折返：[3,5,4,1,2]的日Spearman为-0.6，与+1方向不一致。
        zigzag = self.returns_frame({"2026-01-01": {1: 3.0, 2: 5.0, 3: 4.0, 4: 1.0, 5: 2.0}})
        result = group_quality_statistics(zigzag, self.directions(1), 5)
        self.assertLess(result["group_monotonicity"][0], 0.0)
        self.assertAlmostEqual(result["monotonic_days_ratio"][0], 0.0)
        self.assertAlmostEqual(result["group_spread_positive_days"][0], 0.0)

    # 【非核心】退化三组时五组单调性为null，端点差为正日比例按三组端点计算。
    def test_sign_mode_null_monotonicity(self):
        """sign模式单调性返回null，正负组差与方向一致时比例可算。"""
        frame = self.returns_frame({"2026-01-01": {1: 1.0, 2: 0.0, 3: 3.0},
                                    "2026-01-02": {1: 2.0, 3: 4.0}})
        result = group_quality_statistics(frame, self.directions(1), 3)
        self.assertIsNone(result["group_monotonicity"][0])
        self.assertIsNone(result["monotonic_days_ratio"][0])
        # 两日正负组差均为2，方向+1时为正比例1。
        self.assertAlmostEqual(result["group_spread_positive_days"][0], 1.0)

    # 【非核心】极端分位阈值只由训练期拟合，验证期极端值不改变阈值。
    def test_extreme_thresholds_frozen(self):
        """fit_group_boundaries的P5/P95来自隔离边界以前的训练样本。"""
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "pressure.parquet"
            boundary = int(dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
            pl.DataFrame({
                "date": ["2026-05-31"] * 5 + ["2026-06-01"],
                "timestamp_ns": [boundary - seconds * 1_000_000_000 for seconds in [304, 303, 302, 301, 300, -1]],
                "pressure": [1.0, 2.0, 3.0, 4.0, 5.0, 10000.0],
            }).write_parquet(filename)
            result = fit_group_boundaries(filename, {
                "dates": ["2026-05-31", "2026-06-01"], "train_end": "2026-05-31",
                "purge_ns": 300_000_000_000, "groups": 5,
             "factor_lag_ns": 0})
            # 训练值[1,2,3,4]（304至301秒在隔离边界前），P5/P95只在这四个值之间。
            self.assertAlmostEqual(result["p05"], 1.15)
            self.assertAlmostEqual(result["p95"], 3.85)
        table = pl.DataFrame({"value": [0.0, 0.5, 1.9, 2.0, 3.0], "target": [-4.0, -3.0, 6.0, 8.0, 9.0],
                              "label_valid": [True] * 5})
        day = extreme_tail_days(table, "f", "2026-06-01", "validation", 1_000_000_000, "all", 0.05, 1.95)
        # 高于1.95的均值(8+9)/2，低于0.05的均值-4。
        self.assertAlmostEqual(day["extreme_spread_bps"], 8.5 - (-4.0))
        self.assertEqual(day["high_rows"], 2)
        self.assertEqual(day["low_rows"], 1)
        missing_low = extreme_tail_days(table.filter(pl.col("value") >= 0.5), "f", "2026-06-02",
                                        "validation", 1_000_000_000, "all", 0.05, 1.95)
        self.assertIsNone(missing_low["extreme_spread_bps"])

    # 【非核心】常数因子与样本不足时不产生错误斜率。
    def test_constant_factor_ols(self):
        """训练方差为零或样本不足时beta与R2为null。"""
        empty = single_factor_ols({})
        self.assertIsNone(empty["beta_1std_bps"])
        single = single_factor_ols({"train": (1, 1.0, 1.0, 2.0, 4.0, 2.0)})
        self.assertIsNone(single["beta_1std_bps"])
        constant = single_factor_ols({"train": (5, 25.0, 125.0, 10.0, 200.0, 50.0)})
        self.assertIsNone(constant["beta_1std_bps"])

    # 【非核心】OLS拟合与样本外评价时间严格分离，基准为训练期常数均值。
    def test_ols_train_only_fit(self):
        """训练y=3x确定斜率，验证同构R2为1，测试反向关系R2大幅为负。"""
        xs = np.arange(1, 51, dtype=float)
        ys = 3.0 * xs
        train = (50, xs.sum(), (xs ** 2).sum(), ys.sum(), (ys ** 2).sum(), (xs * ys).sum())
        same = (50, xs.sum(), (xs ** 2).sum(), ys.sum(), (ys ** 2).sum(), (xs * ys).sum())
        opposite = -ys
        test = (50, xs.sum(), (xs ** 2).sum(), opposite.sum(), (opposite ** 2).sum(), (xs * opposite).sum())
        result = single_factor_ols({"train": train, "validation": same, "test": test})
        self.assertAlmostEqual(result["beta_1std_bps"], 3.0 * np.std(xs), places=9)
        self.assertAlmostEqual(result["validation_oos_r2"], 1.0, places=9)
        self.assertLess(result["test_oos_r2"], 0.0)
        # 验证期改为常数且等于训练均值时，分母退化返回null而非0。
        flat = np.full(50, ys.mean())
        constant = (50, xs.sum(), (xs ** 2).sum(), flat.sum(), (flat ** 2).sum(), (xs * flat).sum())
        result = single_factor_ols({"train": train, "validation": constant})
        self.assertIsNone(result["validation_oos_r2"])

    # 【非核心】完整入口同时输出分组质量、极端分位与单因子OLS表。
    def test_entrypoint_outputs(self):
        """多日正相关因子走真实评估，新指标列与文件正确生成。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "test_factor"
            dates = ["2026-07-12", "2026-07-13", "2026-07-14"]
            frames = []
            for offset, date in enumerate(dates):
                keys = {"symbol": ["BTCUSDT"] * 100, "date": [date] * 100,
                        "timestamp_ns": int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000 + ((np.arange(100, dtype=np.int64) + offset * 100) * 1_000_000_000)}
                factor = np.arange(100, dtype=float)
                frames.append(pl.DataFrame({**keys, name: factor}))
                labels = pl.DataFrame({**keys, "label_valid": [True] * 100,
                                       "label_end_timestamp_ns": int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000 + ((np.arange(100, dtype=np.int64) + offset * 100)
                                       * 1_000_000_000 + 1_000_000_000),
                                       "y_mid_1s_bps": factor * 2.0})
                write_parquet(labels.lazy(), root / f"labels/test/horizon=1s/{date}.parquet",
                              {"kind": "labels", "dataset": {"dates": dates, "symbol": "BTCUSDT"},
                               "horizon_ns": 1_000_000_000})
            write_parquet(pl.concat(frames).lazy(), root / f"factors/test/{name}.parquet",
                          {"kind": "factor", "dataset": {"dates": dates, "symbol": "BTCUSDT"},
                           "definition": {"dates": dates}})
            config = {"stage": "evaluation", "factor_root": str(root / "factors"), "factor_version": "test",
                      "label_root": str(root / "labels"), "label_version": "test", "dates": dates,
                      "factors": [name], "horizons_ns": [1_000_000_000], "groups": 5,
                      "output": str(root / "results"), "train_end": dates[0], "validation_end": dates[1],
                      "workers": 1, "purge_ns": 300_000_000_000, "output_interval_ns": 1_000_000_000,
                      "factor_batch_size": 6, "ols_factors": [name], "factor_lag_ns": 0, "validation_dev_end": dates[1]}
            (root / "evaluation.yaml").write_text(yaml.safe_dump(config))
            run_evaluation(root / "evaluation.yaml")
            summary = pl.read_parquet(root / f"results/{name}/summary.parquet")
            train = summary.filter((pl.col("split") == "train") & (pl.col("sampling") == "all"))
            # 完美递增五组：单调性1，方向一致日比例1，端点差为正比例1。
            self.assertAlmostEqual(train["group_monotonicity"][0], 1.0)
            self.assertAlmostEqual(train["monotonic_days_ratio"][0], 1.0)
            self.assertAlmostEqual(train["group_spread_positive_days"][0], 1.0)
            # 均匀0-99因子P5=4.95、P95=94.05：高尾均值194、低尾均值4，逐日差190。
            self.assertAlmostEqual(train["extreme_5pct_spread_bps"][0], 190.0, places=6)
            self.assertEqual(train["extreme_valid_days"][0], 1)
            extremes = pl.read_parquet(root / f"results/{name}/extreme_returns.parquet")
            self.assertEqual(extremes.height, len(dates) * 2)
            ols = pl.read_parquet(root / f"results/{name}/single_factor_ols.parquet")
            self.assertAlmostEqual(ols["beta_1std_bps"][0], 2.0 * np.std(np.arange(100, dtype=float)), places=9)
            self.assertAlmostEqual(ols["val_dev_oos_r2"][0], 1.0, places=9)
            # 测试期同样为完美线性关系，样本外R2也为1。
            self.assertAlmostEqual(ols["test_oos_r2"][0], 1.0, places=9)
            self.assertEqual(json.loads((root / f"results/{name}/completed.json").read_text())["definition_version"],
                             "coarse_windows_log_bps_seven_horizons_v3")


if __name__ == "__main__":
    unittest.main()
