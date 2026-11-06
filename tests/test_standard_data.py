"""验证官方逐笔、质量标签、二十五档保留及完整发布。

小型真实压缩文件通过生产API读取，覆盖重复、冲突、未知字段和时间口径。
"""

import datetime as dt
from pathlib import Path
import tempfile
import unittest
import zipfile

import polars as pl
from polars.testing import assert_frame_equal
import zstandard

from mm_research.data.sources import load_official_trades
from mm_research.data.standard import build_standard_day, read_standard_metadata, run_standard_production
from mm_research.data.trades import prepare_official_trades

DATE = "2026-07-12"
START_US = int(dt.datetime(2026, 7, 12, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000


# ==================== 确定性压缩源文件 ====================
# 写出实际会被Polars读取的压缩CSV，保留测试给定的行顺序。
def write_compressed_csv(filename: Path, table: pl.DataFrame) -> None:
    """输入文件位置与表，写出Zstandard CSV，供生产读取路径验收。"""
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_bytes(zstandard.ZstdCompressor().compress(table.write_csv().encode()))


# 构造带完整二十五档且具有重复时点的原始快照。
def make_snapshots() -> pl.DataFrame:
    """返回四行源顺序快照，包含相同时刻、已知缺口、乱序和最深档缺失。"""
    values = {
        "exchange": ["binance-futures"] * 4, "symbol": ["BTCUSDT"] * 4,
        "timestamp": [START_US + offset - 1000 for offset in [100_000, 100_000, 1_500_000, 1_400_000]],
        "local_timestamp": [START_US + offset for offset in [100_000, 100_000, 1_500_000, 1_400_000]],
    }
    for level in range(25):
        for raw_side, sign in [("bids", -1), ("asks", 1)]:
            values[f"{raw_side}[{level}].price"] = [100 + sign * (level + 1) * 0.1] * 4
            values[f"{raw_side}[{level}].amount"] = [0.001 * (level + 1), 0.002, 0.003, 0.004]
    values["asks[24].amount"][-1] = None
    return pl.DataFrame(values)


# 以真实ZIP布局保存官方格式成交，接收时间不属于该来源。
def write_official_archive(filename: Path, ids: list[int], times_ms: list[int]) -> None:
    """输入官方编号及毫秒时间，生成价格数量方向确定的普通成交ZIP。"""
    table = pl.DataFrame({
        "id": ids, "price": [100.0] * len(ids), "qty": [0.001] * len(ids),
        "quote_qty": [0.1] * len(ids), "time": times_ms, "is_buyer_maker": [False] * len(ids),
    })
    filename.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("trades.csv", table.write_csv())


# ==================== 上游标准表验收 ====================
class StandardDataTests(unittest.TestCase):
    """通过实际压缩源文件验证官方成交与快照生产接口。"""

    # 两份YAML所属阶段不同，传错入口时明确说明应使用的配置。
    def test_reject_features_config(self):
        """检查Features配置不能被Data标准化入口执行。"""
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / "features.yaml"
            config_file.write_text("stage: features_targets\n")
            with self.assertRaisesRegex(ValueError, "Data标准化"):
                run_standard_production(config_file)

    # 官方编号跳跃直接保留，重复副本合并且冲突可追溯。
    def test_official_duplicates_and_number_gaps(self):
        """检查毫秒转换、来源行号、重复冲突与确定性。"""
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "trades.zip"
            write_official_archive(filename, [1, 1, 3, 3, None, None], [START_US // 1000 + 1] * 6)
            raw = load_official_trades(filename, "official", 0.1, 0.001)
            raw = raw.with_columns(
                pl.when(pl.col("source_row") == 4).then(101.0).otherwise(pl.col("price")).alias("price"),
            )
            result = prepare_official_trades(raw, DATE)
            self.assertEqual(result.height, 4)
            self.assertEqual(result["trade_id"].null_count(), 2)
            first = result.filter(pl.col("trade_id") == 1).row(0, named=True)
            self.assertEqual(first["exchange_timestamp_ns"], (START_US + 1000) * 1000)
            self.assertEqual(first["side"], "buy")
            self.assertEqual(first["source_row"], 1)
            self.assertIn("duplicate_collapsed", first["quality_flags"])
            conflict = result.filter(pl.col("trade_id") == 3).row(0, named=True)
            self.assertEqual(conflict["price"], 100.0)
            self.assertIn("content_conflict", conflict["quality_flags"])
            self.assertFalse(any("id_gap" in flag for flags in result["quality_flags"] for flag in flags))
            assert_frame_equal(result, prepare_official_trades(raw, DATE), check_exact=True)

    # 异常字段保留原值并附加标签，未知方向不能默认成买入。
    def test_unknown_and_invalid_fields(self):
        """检查方向、非正价量、乱序时间及越日记录。"""
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "trades.zip"
            with zipfile.ZipFile(filename, "w") as archive:
                archive.writestr("trades.csv", "id,price,qty,quote_qty,time,is_buyer_maker\n"
                                 f"1,100,0.001,0.1,{START_US // 1000 + 1},true\n"
                                 f"3,0,-1,0,{START_US // 1000},\n"
                                 f",100,0.001,0.1,{START_US // 1000 - 1},false\n")
            result = prepare_official_trades(load_official_trades(filename, "official", 0.1, 0.001), DATE)
            invalid = result.filter(pl.col("trade_id") == 3).row(0, named=True)
            self.assertIsNone(invalid["side"])
            for flag in ["invalid_side", "invalid_price", "invalid_quantity", "exchange_time_backwards"]:
                self.assertIn(flag, invalid["quality_flags"])
            self.assertEqual(result.filter(pl.col("trade_id") == 1)["side"][0], "sell")
            self.assertIn("outside_trade_day", result.filter(pl.col("trade_id").is_null())["quality_flags"][0])

    # 无需采集成交CSV，直接生产两份交易所时钟的标准表。
    def test_day_publication_depth_clock_and_cache(self):
        """验证完整25档、原始行序、时间模型、缓存和失败清理。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_compressed_csv(root / f"raw/{DATE}/snapshots.zst", make_snapshots())
            write_official_archive(root / f"raw/{DATE}/trades.zip", [1, 3], [START_US // 1000 + 1] * 2)
            config = {
                "version": "market_v2", "output_root": "standard",
                "snapshots": "raw/{date}/snapshots.zst", "trades": "raw/{date}/trades.zip",
                "tick_size": 0.1, "lot_size": 0.001, "snapshot_silence_ns": 1_000_000_000,
                "confirmed_snapshot_gaps": {DATE: [[(START_US + 100_000) * 1000, (START_US + 1_500_000) * 1000]]},
            }
            summary = build_standard_day(root, config, DATE)
            folder = root / f"standard/market_v2/date={DATE}"
            self.assertEqual(sorted(path.name for path in folder.iterdir()), ["snapshots.parquet", "trades.parquet"])
            snapshots = pl.read_parquet(folder / "snapshots.parquet")
            self.assertEqual(snapshots.height, 4)
            self.assertEqual(snapshots["source_row"].to_list(), [1, 2, 3, 4])
            self.assertIsNone(snapshots["ask_quantity_25"][-1])
            self.assertIn("gap_before_confirmed", snapshots["quality_flags"][2])
            self.assertIn("source_received_time_backwards", snapshots["quality_flags"][3])
            self.assertIn("incomplete_depth", snapshots["quality_flags"][3])
            self.assertNotIn("received_timestamp_ns", snapshots.columns)
            self.assertNotIn("received_timestamp_ns", pl.read_parquet_schema(folder / "trades.parquet"))
            definition = read_standard_metadata(folder / "snapshots.parquet")["definition"]
            self.assertEqual(definition["schema_version"], 2)
            self.assertEqual(definition["snapshot_depth"], 25)
            self.assertEqual(definition["market_clock"], "exchange_timestamp_ns")
            self.assertIn("at replay", definition["receipt_time_model"])
            modified = (folder / "trades.parquet").stat().st_mtime_ns
            self.assertEqual(build_standard_day(root, config, DATE), summary)
            self.assertEqual((folder / "trades.parquet").stat().st_mtime_ns, modified)
            with self.assertRaises(ValueError):
                build_standard_day(root, config | {"tick_size": 0.2}, DATE)
            with zipfile.ZipFile(root / f"raw/{DATE}/trades.zip", "w") as archive:
                archive.writestr("empty.txt", "bad input")
            with self.assertRaises(ValueError):
                build_standard_day(root, config | {"version": "failed"}, DATE)
            self.assertFalse((root / f"standard/failed/date={DATE}").exists())
            self.assertEqual(list((root / "standard/failed").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
