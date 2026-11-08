"""独立矩阵参考核对流式统计、三种线性拟合及样本外打分。"""

import unittest
from pathlib import Path
import tempfile
import datetime as dt

import polars as pl
import yaml

from mm_research.data.storage import write_parquet
from mm_research.models.runner import run_linear_baseline
from mm_research.models.dataset import RAW_COLUMNS
from mm_research.factors.dataset import factor_module

import numpy as np

from mm_research.models.linear import fit_linear_model, linear_statistics, merge_statistics, score_linear_model


class LinearModelTests(unittest.TestCase):
    """避免按日分块和原单位系数导出改变模型定义。"""

    # 【非核心】分块合并应等价于整段中心化，OLS与独立最小二乘一致。
    def test_partition_and_ols(self):
        """不同量纲输入和非零截距用于检验标准化及导出的原单位系数。"""
        random = np.random.default_rng(91)
        matrix = random.normal(size=(300, 3)) * [0.01, 1, 10000] + [2, -3, 1e6]
        target = matrix @ np.array([0.3, -0.8, 0.0002]) + 2
        full = linear_statistics(matrix, target)
        merged = merge_statistics(linear_statistics(matrix[:130], target[:130]), linear_statistics(matrix[130:], target[130:]))
        np.testing.assert_allclose(merged.xx, full.xx, rtol=1e-10, atol=1e-5)
        np.testing.assert_allclose(merged.xy, full.xy, rtol=1e-10, atol=1e-8)
        model = fit_linear_model(merged, "ols", 0)
        prediction = matrix @ model["coefficients"] + model["intercept"]
        np.testing.assert_allclose(prediction, target, rtol=1e-10, atol=1e-9)
        self.assertAlmostEqual(score_linear_model(full, model)["ic"], 1)

    # 【非核心】用三个隔离日期执行完整入口，检查选列、训练、预测与C++系数导出。
    def test_model_pipeline_entry(self):
        """人工数据仅用于接口测试，评价汇总只提供训练期候选。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = ["2026-07-01", "2026-07-02", "2026-07-03"]
            random = np.random.default_rng(93)
            rows = 600
            timestamps = []
            for date in dates:
                start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
                timestamps.extend(start + np.arange(rows) * 100_000_000)
            keys = pl.DataFrame({"symbol": ["BTCUSDT"] * (rows * 3),
                                 "date": [date for date in dates for _ in range(rows)], "timestamp_ns": timestamps})
            dataset = {"dates": dates, "symbol": "BTCUSDT"}
            values = keys.with_columns([pl.Series(name, random.normal(size=rows * 3)) for name in RAW_COLUMNS])
            for name in RAW_COLUMNS:
                write_parquet(values.lazy().select("symbol", "date", "timestamp_ns", name), root / "features" / f"{name}.parquet",
                              {"kind": "feature", "dataset": dataset, "definition": {}})
            names = ["book_obi_l1_mean_10s", "book_obi_l1_mean_diff_10s_50s"]
            for batch, name in zip(["gen001", "gen002"], names):
                expression = factor_module(batch).factor_expressions({"min_fraction": 0.8})[name]
                write_parquet(values.lazy().select("symbol", "date", "timestamp_ns", expression.alias(name)),
                              root / "factors" / f"{name}.parquet", {"kind": "factor", "dataset": dataset, "definition": {}})
            for date in dates:
                label = values.filter(pl.col("date") == date).select("timestamp_ns").with_columns(
                    pl.lit(True).alias("label_valid"),
                    (pl.col("timestamp_ns") + 1_000_000_000).alias("label_end_timestamp_ns"),
                    pl.Series("y_mid_1s_bps", values.filter(pl.col("date") == date)["imbalance_1"] * 0.1),
                )
                write_parquet(label.lazy(), root / "labels/horizon=1s" / f"{date}.parquet",
                              {"kind": "labels", "dataset": dataset, "horizon_ns": 1_000_000_000})
            (root / "evaluation").mkdir()
            pl.DataFrame({"factor": names, "split": ["train"] * 2, "sampling": ["nonoverlap"] * 2,
                          "horizon_ns": [1_000_000_000] * 2, "ic_mean": [0.1, 0.05]}).write_parquet(root / "evaluation/summary.parquet")
            config = {
                "stage": "models", "feature_root": str(root), "feature_version": "features",
                "factor_root": str(root), "factor_version": "factors", "label_root": str(root), "label_version": "labels",
                "evaluation_root": str(root / "evaluation"), "dates": dates, "train_end": dates[0], "validation_end": dates[1],
                "batches": ["gen001", "gen002"], "min_fraction": 0.8, "factors_per_batch": 1, "realtime_per_batch": 1,
                "sample_stride_ns": 1_000_000_000, "ridge_alphas": [0.01], "lasso_alphas": [0.01],
                "workers": 1, "output": str(root / "results"), "purge_ns": 300_000_000_000,
            }
            filename = root / "models.yaml"
            filename.write_text(yaml.safe_dump(config))
            report = run_linear_baseline(filename)
            self.assertEqual(set(report["selected_metrics_nonoverlap"]), {"history", "raw", "factors", "realtime"})
            self.assertTrue((root / "results/realtime_model.txt").exists())
            self.assertTrue((root / f"results/predictions/{dates[-1]}.parquet").exists())

    # 【非核心】高度相关的压力窗口仍须达到Lasso最优性条件。
    def test_lasso_collinear_windows(self):
        """相近窗口的近共线输入通过独立KKT残差检查。"""
        random = np.random.default_rng(94)
        common = random.normal(size=1000)
        matrix = common[:, None] + random.normal(size=(1000, 6)) * 0.001
        target = common * 0.3 + random.normal(size=1000) * 0.01
        model = fit_linear_model(linear_statistics(matrix, target), "lasso", 0.00001)
        standardized = (matrix - matrix.mean(0)) / matrix.std(0)
        coefficients = np.asarray(model["standardized_coefficients"])
        gradient = standardized.T @ (standardized @ coefficients - (target - target.mean())) / len(target)
        residual = np.where(coefficients != 0, np.abs(gradient + 0.00001 * np.sign(coefficients)),
                            np.maximum(np.abs(gradient) - 0.00001, 0))
        self.assertLessEqual(float(residual.max()), 1.0001e-7)

    # 【非核心】Ridge解与独立标准化公式一致，Lasso大惩罚只保留训练均值。
    def test_regularization_and_heldout_score(self):
        """验证阶段只打分，保持训练参数及标准化冻结。"""
        random = np.random.default_rng(92)
        matrix = random.normal(size=(200, 4))
        target = matrix[:, 0] + random.normal(size=200) * 0.2
        train = linear_statistics(matrix[:120], target[:120])
        model = fit_linear_model(train, "ridge", 0.1)
        standardized = (matrix[:120] - matrix[:120].mean(0)) / matrix[:120].std(0)
        expected = np.linalg.solve(standardized.T @ standardized / 120 + 0.1 * np.eye(4),
                                   standardized.T @ (target[:120] - target[:120].mean()) / 120)
        np.testing.assert_allclose(model["standardized_coefficients"], expected, atol=1e-12)
        heldout = linear_statistics(matrix[120:], target[120:])
        prediction = matrix[120:] @ model["coefficients"] + model["intercept"]
        self.assertAlmostEqual(score_linear_model(heldout, model)["mse_bps2"], np.mean((prediction - target[120:]) ** 2))
        constant = fit_linear_model(train, "lasso", 100)
        self.assertTrue(all(value == 0 for value in constant["coefficients"]))
        self.assertIsNone(score_linear_model(heldout, constant)["ic"])


if __name__ == "__main__":
    unittest.main()
