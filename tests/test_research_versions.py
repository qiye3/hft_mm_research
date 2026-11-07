"""验证单特征文件、增量更新、独立标签与缺失采样。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import polars as pl
from test_standard_data import make_snapshots, write_compressed_csv, write_official_archive, START_US, DATE
from mm_research.data.standard import build_standard_day
from mm_research.data.dataset import build_features, build_labels_day, load_xy
from mm_research.data.storage import read_metadata
from mm_research.features.base import build_base_features
from mm_research.features.validation import validate_feature_intermediates


class ResearchVersionTests(unittest.TestCase):
    """检查当前单变量研究格式。"""

    # 【非核心】检查单变量布局，增加组时复用基础列，标签保持独立。
    def test_single_series_incremental_and_labels(self):
        """小型网格通过正式生产入口写出并按时间键读取X/y。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_standard_fixture(root)
            config = fixture_config(root)
            grid = pl.DataFrame({"timestamp_ns": [START_US * 1000 + index * 100_000_000 for index in range(180)]})
            with patch("mm_research.features.snapshot.make_day_grid", return_value=grid):
                build_features(root, config, ["base", "book"])
            features = root / "features/test_features"
            filename = features / "imbalance_5.parquet"
            original = filename.read_bytes()
            with patch("mm_research.data.dataset.build_base_features", side_effect=AssertionError("重复采样")):
                build_features(root, config, ["base", "trades", "returns"])
            self.assertEqual(filename.read_bytes(), original)
            self.assertEqual(pl.read_parquet(filename).columns, ["symbol", "date", "timestamp_ns", "imbalance_5"])
            self.assertFalse(list(features.glob("2026-*.parquet")))
            label = build_labels_day(root, config, DATE, 1_000_000_000)
            modified = label.stat().st_mtime_ns
            self.assertEqual(build_labels_day(root, config, DATE, 1_000_000_000), label)
            self.assertEqual(label.stat().st_mtime_ns, modified)
            matrix, target, index = load_xy(features, label.parent, [DATE], ["imbalance_5", "trade_imbalance_1s"])
            self.assertEqual(matrix.shape[1], 2)
            self.assertGreater(len(target), 0)
            self.assertEqual(index.height, len(target))
            self.assertEqual(read_metadata(filename)["kind"], "feature")

    # 【非核心】检查写盘失败保留已有文件，随后可显式重试。
    def test_failed_feature_write_and_retry(self):
        """对原子发布注入磁盘错误，核对已有结果完整性。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_standard_fixture(root)
            config = fixture_config(root)
            grid = pl.DataFrame({"timestamp_ns": [START_US * 1000 + index * 100_000_000 for index in range(100)]})
            with patch("mm_research.features.snapshot.make_day_grid", return_value=grid):
                build_features(root, config, ["base", "book"])
            filename = root / "features/test_features/imbalance_5.parquet"
            original = pl.read_parquet(filename)
            config["feature_groups"] = {"book": "v2"}
            with patch.object(pl.LazyFrame, "collect_batches", side_effect=OSError("模拟写入失败")):
                with self.assertRaises(OSError):
                    build_features(root, config, ["base", "book"], rebuild=True)
            self.assertTrue(pl.read_parquet(filename).equals(original))
            build_features(root, config, ["base", "book"], rebuild=True)
            self.assertTrue(pl.read_parquet(filename).equals(original))

    # 【非核心】检查异常采样置空，正常无成交窗口为零，且独立参考计算一致。
    def test_standard_alignment_and_null_observations(self):
        """读取小型标准日，核对初始未覆盖窗口与盘口对齐。"""
        with tempfile.TemporaryDirectory() as directory:
            folder = prepare_standard_fixture(Path(directory))
            grid = pl.DataFrame({"timestamp_ns": [START_US * 1000 + index * 100_000_000 for index in range(180)]})
            with patch("mm_research.features.snapshot.make_day_grid", return_value=grid):
                base = build_base_features(folder, DATE, 100_000_000, 0.001, 1_000_000_000, 20_000_000)
            self.assertIsNone(base["mid_price"][0])
            self.assertIsNone(base["buy_volume"][0])
            self.assertEqual(base["buy_volume"][2], 0.002)
            self.assertEqual(base["buy_volume"][10], 0)
            self.assertNotIn("trade_segment_id", base.columns)
            report = validate_feature_intermediates(base, folder, DATE, 20_000_000, 100_000_000, 0.001, 1_000_000_000)
            self.assertEqual(report["visible_trades"], 2)


# 【非核心】构造唯一Features/Targets生产入口使用的最小配置。
def fixture_config(root: Path) -> dict:
    """返回单日分组配置，输出仍为标准逐日布局。"""
    return {
        "stage": "features_targets", "standard_root": str(root / "standard"),
        "standard_version": "market_v2", "market_data_latency_ns": 20_000_000,
        "dates": [DATE], "feature_groups": {"base": "v3", "book": "v2", "trades": "v2", "returns": "v2"},
        "feature_root": str(root / "features"), "feature_version": "test_features",
        "label_root": str(root / "labels"), "label_version": "test_labels",
        "tick_size": 0.1, "lot_size": 0.001, "horizons_ns": [1_000_000_000],
        "rolling_min_samples": {"1s": 8, "5s": 40}, "interval_ns": 100_000_000, "max_book_age_ns": 1_000_000_000,
    }


# 【非核心】制作可通过正式生产接口读取的完整25档快照和官方成交ZIP。
def prepare_standard_fixture(root: Path) -> Path:
    """返回标准版本目录，所选日期快照每100ms到达且成交编号有正常空缺。"""
    source = make_snapshots().head(1)
    snapshots = pl.concat([
        source.with_columns(
            pl.lit(START_US + 100_000 + index * 100_000).alias("timestamp"),
            pl.lit(START_US + 102_000 + index * 100_000).alias("local_timestamp"),
        ) for index in range(150)
    ])
    write_compressed_csv(root / "raw/snapshots.zst", snapshots)
    write_official_archive(root / "raw/trades.zip", [1, 3], [START_US // 1000 + 100, START_US // 1000 + 180])
    config = {
        "version": "market_v2", "output_root": "standard", "snapshots": "raw/snapshots.zst",
        "trades": "raw/trades.zip", "tick_size": 0.1, "lot_size": 0.001,
        "snapshot_silence_ns": 1_000_000_000, "confirmed_snapshot_gaps": {},
    }
    build_standard_day(root, config, DATE)
    return root / "standard/market_v2"


if __name__ == "__main__":
    unittest.main()
