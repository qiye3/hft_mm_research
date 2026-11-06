"""验证独立特征与标签版本、bps 单位以及快照快捷入口的因果边界。

人工序列用于验证三个目标单位和多周期日界，临时工件用于检查独立重建与错配拒绝。
小型快照和成交输入通过真实 Polars 读取路径生成网格，检查时间对齐与缺口标记。
"""

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from mm_research.data.sources import DAY_NS
from mm_research.features.base import build_base_features
from mm_research.features.trades import build_trade_bins
from mm_research.data.standard import build_standard_day
from mm_research.features.validation import validate_feature_intermediates
from test_standard_data import make_snapshots, write_compressed_csv, write_official_archive, START_US, DATE
from mm_research.data.dataset import (
    build_features_day, load_xy, read_artifact_metadata, run_dataset_production, write_partition,
)
from mm_research.features.transform import GROUPS, build_feature_group
from mm_research.targets.midprice import build_midprice_labels


# 【非核心】构造具有已知未来价格差的单日基础表。
def make_base(count: int = 120, start: int = 0) -> pl.DataFrame:
    """输入长度和起始纳秒，返回包含完整十档与恒定成交量的确定性基础表。"""
    values = {
        "symbol": ["BTCUSDT"] * count, "date": ["2026-07-12"] * count,
        "timestamp_ns": start + np.arange(count, dtype=np.int64) * 100_000_000,
        "last_available_timestamp_ns": start + np.arange(count, dtype=np.int64) * 100_000_000,
        "book_valid": [True] * count, "flow_valid": [True] * count,
        "book_segment_id": [1] * count, "trade_segment_id": [1] * count,
        "mid_price": 60000 + np.arange(count) * 0.1,
        "spread": [0.2] * count,
        "buy_volume": [0.002] * count, "sell_volume": [0.001] * count,
    }
    for level in range(1, 11):
        values[f"bid_price_{level}"] = [60000 - level * 0.1] * count
        values[f"ask_price_{level}"] = [60000 + level * 0.1] * count
        values[f"bid_quantity_{level}"] = [2.0] * count
        values[f"ask_quantity_{level}"] = [1.0] * count
    return pl.DataFrame(values)


class ResearchVersionTests(unittest.TestCase):
    """针对版本隔离、单位和本地可见时间执行回归。"""

    # 【非核心】三个单位对应同一价格变化，多周期严格检查未来时刻和跨日边界。
    def test_target_units_horizons_and_day_boundary(self):
        """用可手算的价格序列核对 bps、Ticks、USDT，并保证标签独立于特征预热。"""
        base = make_base()
        for horizon, suffix, steps in [(100_000_000, "100ms", 1), (500_000_000, "500ms", 5),
                                       (1_000_000_000, "1s", 10), (5_000_000_000, "5s", 50)]:
            labels = build_midprice_labels(base, 0.1, 100_000_000, horizon)
            self.assertAlmostEqual(labels[f"y_mid_{suffix}_usdt"][0], steps * 0.1)
            self.assertAlmostEqual(labels[f"y_mid_{suffix}_ticks"][0], steps)
            self.assertAlmostEqual(labels[f"y_mid_{suffix}_bps"][0], steps * 0.1 / 60000 * 10000)
            self.assertTrue(labels["label_valid"][0])
            self.assertFalse(labels.tail(steps)["label_valid"].any())
        boundary = make_base(40, DAY_NS - 2_000_000_000)
        labels = build_midprice_labels(boundary, 0.1, 100_000_000, 1_000_000_000)
        self.assertFalse(labels["label_valid"][15])
        self.assertTrue(labels["label_valid"][20])
        missing = base.with_columns(pl.when(pl.col("timestamp_ns") == 6_500_000_000)
                                    .then(None).otherwise(pl.col("mid_price")).alias("mid_price"))
        self.assertFalse(build_midprice_labels(missing, 0.1, 100_000_000)["label_valid"][60])
        config = {"tick_size": 0.1, "lot_size": 0.001}
        damaged_flow = base.with_columns(pl.lit(False).alias("flow_valid"))
        book = build_feature_group("book", damaged_flow, config)
        trades = build_feature_group("trades", damaged_flow, config)
        self.assertTrue(book["book_feature_valid"][0])
        self.assertFalse(trades["trade_feature_valid"].any())
        self.assertTrue(build_midprice_labels(damaged_flow, 0.1, 100_000_000)["label_valid"][0])

    # 【非核心】增加组时复用同一日期特征文件里的基础列，标签独立复用。
    def test_incremental_groups_and_independent_labels(self):
        """验证自动base依赖、只计算新增组、组版本更新和按组选取X。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_standard_fixture(root)
            config = fixture_config(root)
            config["feature_groups"] = {"book": "v1"}
            config_file = root / "config.yaml"
            config_file.write_text(yaml.safe_dump({"stage": "data_standardize"}))
            with self.assertRaisesRegex(ValueError, "Features/Targets"):
                run_dataset_production(config_file)
            config_file.write_text(yaml.safe_dump(config))
            run_dataset_production(config_file)
            feature_root = root / "features/test_features"
            label_root = root / "labels/test_labels/horizon=1s"
            filename = feature_root / f"{DATE}.parquet"
            label_file = label_root / f"{DATE}.parquet"
            label_time = label_file.stat().st_mtime_ns
            initial = pl.read_parquet(filename)
            self.assertEqual(set(read_artifact_metadata(filename)["groups"]), {"base", "book"})
            self.assertEqual(list(filename.parent.iterdir()), [filename])
            config["feature_groups"] = {"trades": "v1", "returns": "v1"}
            config_file.write_text(yaml.safe_dump(config))
            # 成功增加两组时，重新读取原始行情会使该测试直接失败。
            with patch("mm_research.data.dataset.build_base_features", side_effect=AssertionError("重复计算base")):
                run_dataset_production(config_file, only="features")
            result = pl.read_parquet(filename)
            for column in initial.columns:
                self.assertTrue(initial[column].equals(result[column]))
            self.assertEqual(label_file.stat().st_mtime_ns, label_time)
            features, targets, index = load_xy(feature_root, label_root)
            self.assertEqual(features.shape[1], 16)
            self.assertGreater(index.height, 0)
            self.assertTrue(np.isfinite(targets).all())
            with patch.object(Path, "glob", side_effect=AssertionError("指定日期时重复发现目录")):
                book_only, _, _ = load_xy(feature_root, label_root, dates=[DATE], groups=["book"])
            self.assertEqual(book_only.shape[1], 7)
            self.assertGreater(book_only.shape[0], features.shape[0])
            feature_time = filename.stat().st_mtime_ns
            config["horizons_ns"].append(500_000_000)
            config_file.write_text(yaml.safe_dump(config))
            with patch("mm_research.data.dataset.standard_sources", side_effect=AssertionError("标签不应读取标准行情")):
                run_dataset_production(config_file, only="labels")
            self.assertEqual(filename.stat().st_mtime_ns, feature_time)
            self.assertEqual(label_file.stat().st_mtime_ns, label_time)
            with patch.dict(GROUPS["book"], {"version": "v2"}):
                config["feature_groups"] = {"book": "v2"}
                config_file.write_text(yaml.safe_dump(config))
                run_dataset_production(config_file, only="features")
                self.assertEqual(read_artifact_metadata(filename)["groups"]["book"]["version"], "v2")
                load_xy(feature_root, label_root, groups=["book"])
            self.assertEqual(label_file.stat().st_mtime_ns, label_time)
            with self.assertRaisesRegex(ValueError, "过时"):
                load_xy(feature_root, label_root)
            config["feature_groups"] = {"book": "v1"}
            config_file.write_text(yaml.safe_dump(config))
            run_dataset_production(config_file, only="features")
            config["market_data_latency_ns"] = 30_000_000
            config_file.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "feature_version"):
                run_dataset_production(config_file)
            metadata = read_artifact_metadata(label_file)
            metadata["input_definition"]["market_data_latency_ns"] = 30_000_000
            write_partition(pl.read_parquet(label_file), label_file, metadata)
            with self.assertRaisesRegex(ValueError, "基础组或来源"):
                load_xy(feature_root, label_root)

    # 【非核心】标签命令只读取既有特征，缺少base产物时直接失败。
    def test_labels_require_existing_features(self):
        """检查标签入口不会隐式创建特征或读取标准行情。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture_config(root)
            config_file = root / "config.yaml"
            config_file.write_text(yaml.safe_dump(config))
            with patch("mm_research.data.dataset.build_features_day", side_effect=AssertionError("隐式生产")):
                with self.assertRaises(FileNotFoundError):
                    run_dataset_production(config_file, only="labels")
            self.assertFalse((root / "features").exists())

    # 【非核心】文件发布失败时旧特征和组元数据仍然完整，重试只更新尚未发布的组。
    def test_failed_group_write_and_retry(self):
        """注入写盘失败，检查单文件原子更新与临时产物清理。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_standard_fixture(root)
            config = fixture_config(root)
            filename, _ = build_features_day(root, config, DATE, ["base", "book"])
            original = filename.read_bytes()
            with patch.object(pl.DataFrame, "write_parquet", side_effect=OSError("模拟写入失败")):
                with self.assertRaises(OSError):
                    build_features_day(root, config, DATE, ["base", "trades"])
            self.assertEqual(filename.read_bytes(), original)
            self.assertEqual(list(filename.parent.iterdir()), [filename])
            with patch("mm_research.data.dataset.build_base_features", side_effect=AssertionError("重复计算base")):
                build_features_day(root, config, DATE, ["base", "trades"])
            self.assertEqual(set(read_artifact_metadata(filename)["groups"]), {"base", "book", "trades"})

    # 【非核心】并发增加不同组时，各次更新必须在锁内重新读取最新文件。
    def test_concurrent_group_updates_preserve_both_groups(self):
        """两个调用同时增加trades和returns，最终应保留全部已发布组。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_standard_fixture(root)
            config = fixture_config(root)
            filename, _ = build_features_day(root, config, DATE, ["base", "book"])
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(build_features_day, root, config, DATE, ["base", name])
                           for name in ["trades", "returns"]]
                for future in futures:
                    future.result()
            self.assertEqual(set(read_artifact_metadata(filename)["groups"]), {"base", "book", "trades", "returns"})
            self.assertEqual(list(filename.parent.iterdir()), [filename])

    # 【非核心】无法表示的合法方向数量必须报错，不能转成空值后被当成零成交量。
    def test_trade_lot_overflow_fails(self):
        """通过实际标准表核对整数Lot转换使用严格模式。"""
        with tempfile.TemporaryDirectory() as directory:
            folder = prepare_standard_fixture(Path(directory))
            filename = folder / "trades" / f"{DATE}.parquet"
            trades = pl.read_parquet(filename).with_columns(pl.lit(1e20).alias("quantity"))
            trades.write_parquet(filename)
            with self.assertRaises(pl.exceptions.InvalidOperationError):
                build_base_features(folder, DATE, 100_000_000, 0.001, 1_000_000_000, 20_000_000)

    # 【非核心】检查成交分桶直接使用上游质量标签，并保留异常记录的笔数。
    def test_trade_bins_use_standard_quality_flags(self):
        """给标准成交附加失效标签，核对其不计入主动量且使对应窗口无效。"""
        with tempfile.TemporaryDirectory() as directory:
            folder = prepare_standard_fixture(Path(directory))
            filename = folder / "trades" / f"{DATE}.parquet"
            trades = pl.read_parquet(filename).with_columns(
                pl.lit(["invalid_side"], dtype=pl.List(pl.String)).alias("quality_flags"),
            )
            trades.write_parquet(filename)
            bins = build_trade_bins(folder, DATE, 100_000_000, 0.001, 20_000_000)
            self.assertEqual(bins["trade_count"].sum(), 2)
            self.assertEqual(bins["buy_volume"].sum(), 0)
            self.assertEqual(bins["sell_volume"].sum(), 0)
            self.assertFalse(bins["trade_valid"].any())

    # 【非核心】固定延迟改变可见边界，编号跳跃保持正常，异常盘口仍影响相关区段。
    def test_standard_alignment_and_cross_midnight(self):
        """通过真实标准Parquet验证20ms延迟、同时间戳状态和跨日成交。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = prepare_standard_fixture(root)
            base = build_base_features(folder, DATE, 100_000_000, 0.001, 1_000_000_000, 20_000_000)
            report = validate_feature_intermediates(base, folder, DATE, 20_000_000, 100_000_000, 0.001, 1_000_000_000)
            self.assertEqual(report["visible_trades"], 2)
            self.assertFalse(base["book_valid"][0])
            self.assertFalse(base["book_valid"][1])
            self.assertAlmostEqual(base["mid_price"][2], 100.0)
            self.assertEqual(base["trade_count"][1], 0)
            self.assertEqual(base["trade_count"][2], 2)
            self.assertEqual(base["buy_volume"][2], 0.002)
            self.assertNotIn("trade_sequence_gap", base.columns)
            self.assertNotIn("ofi_bbo", base.columns)
            self.assertNotIn("segment_id", base.columns)
            self.assertFalse(base.filter(pl.col("last_available_timestamp_ns") > pl.col("timestamp_ns")).height)
            tomorrow = "2026-07-13"
            for kind in ["snapshots", "trades"]:
                table = pl.read_parquet(folder / kind / f"{DATE}.parquet")
                table.head(0).write_parquet(folder / kind / f"{tomorrow}.parquet")
            # 前日最后一笔在23:59:59.990成交，延迟后属于次日00:00:00.100网格。
            trades = pl.read_parquet(folder / "trades" / f"{DATE}.parquet")
            late = trades.tail(1).with_columns(
                pl.lit(999).alias("trade_id"),
                pl.lit(START_US * 1000 + DAY_NS - 10_000_000).alias("exchange_timestamp_ns"),
            )
            pl.concat([trades, late], how="vertical_relaxed").write_parquet(folder / "trades" / f"{DATE}.parquet")
            next_base = build_base_features(folder, tomorrow, 100_000_000, 0.001, 1_000_000_000, 20_000_000)
            self.assertEqual(next_base["trade_count"][0], 0)
            self.assertEqual(next_base["trade_count"][1], 1)
            self.assertTrue(next_base["flow_valid"][1])


# 【非核心】构造唯一Features/Targets生产入口使用的最小配置。
def fixture_config(root: Path) -> dict:
    """返回单日分组配置，输出仍为标准逐日布局。"""
    return {
        "stage": "features_targets", "standard_root": str(root / "standard"),
        "standard_version": "market_v2", "market_data_latency_ns": 20_000_000,
        "dates": [DATE], "feature_groups": {"book": "v1"},
        "feature_root": str(root / "features"), "feature_version": "test_features",
        "label_root": str(root / "labels"), "label_version": "test_labels",
        "tick_size": 0.1, "lot_size": 0.001, "horizons_ns": [1_000_000_000],
        "interval_ns": 100_000_000, "max_book_age_ns": 1_000_000_000,
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
