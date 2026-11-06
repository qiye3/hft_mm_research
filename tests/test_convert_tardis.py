"""核对 gzip→NPZ→C++ 引擎的真实格式兼容性及快照/双时钟语义。"""
import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zstandard
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("convert_tardis", ROOT / "scripts/convert_tardis.py")
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)
PROBE = Path(sys.argv[1]).resolve()


class ConversionTests(unittest.TestCase):
    # 【非核心】核对Zstandard和gzip原始文件转换后的事件内容一致。
    def test_zstandard_sources_match_gzip(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            book_csv = (
                "exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount\n"
                "binance-futures,BTCUSDT,100,110,true,bid,99.9,1\n"
                "binance-futures,BTCUSDT,100,110,true,ask,100.1,2\n"
                "binance-futures,BTCUSDT,200,210,false,bid,99.9,0\n"
                "binance-futures,BTCUSDT,200,210,false,bid,99.8,1\n"
            )
            trades_csv = (
                "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
                "binance-futures,BTCUSDT,150,170,1,buy,100.1,1\n"
            )
            sources = [("book", book_csv, ".zstd"), ("trades", trades_csv, ".zst")]
            for name, content, suffix in sources:
                with gzip.open(workspace / f"{name}.csv.gz", "wt") as output:
                    output.write(content)
                (workspace / f"{name}.csv{suffix}").write_bytes(
                    zstandard.ZstdCompressor().compress(content.encode())
                )
            CONVERTER.convert(
                workspace / "book.csv.gz", workspace / "trades.csv.gz", "2026-07-12",
                workspace / "gzip", full_day=False,
            )
            CONVERTER.convert(
                workspace / "book.csv.zstd", workspace / "trades.csv.zst", "2026-07-12",
                workspace / "zstandard", full_day=False,
            )
            with np.load(workspace / "gzip/events-00000.npz") as expected:
                with np.load(workspace / "zstandard/events-00000.npz") as actual:
                    np.testing.assert_array_equal(actual["data"], expected["data"])

    # 【非核心】检查双时钟分片路由与快照清空后的原生回放结果。
    def test_split_clock_shards_and_snapshot_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            book = workspace / "book.csv.gz"
            trades = workspace / "trades.csv.gz"
            with gzip.open(book, "wt") as output:
                output.write("exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount\n")
                for level in range(11):
                    output.write(f"binance-futures,BTCUSDT,100,110,true,bid,{99.9 - level * 0.1:.1f},1\n")
                    output.write(f"binance-futures,BTCUSDT,100,110,true,ask,{100.1 + level * 0.1:.1f},2\n")
                output.write("binance-futures,BTCUSDT,120,130,false,bid,99.9,0\n")
                output.write("binance-futures,BTCUSDT,200,210,true,bid,98.0,3\n")
                output.write("binance-futures,BTCUSDT,200,210,true,ask,102.0,4\n")
            with gzip.open(trades, "wt") as output:
                output.write("exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n")
                output.write("binance-futures,BTCUSDT,90,125,1,buy,100.1,1\n")
                output.write("binance-futures,BTCUSDT,150,170,2,sell,99.8,1\n")
            destination = workspace / "normalized"
            manifest = CONVERTER.convert(book, trades, "2025-02-01", destination, shard_rows=7, full_day=False)
            events = np.concatenate([np.load(destination / shard["file"])["data"] for shard in manifest["shards"]])
            self.assertEqual(events.dtype.itemsize, 64)
            self.assertEqual(len(events), 58)
            self.assertEqual(manifest["logical_events"], 29)
            for mask, field in [(CONVERTER.LOCAL, "local_ts"), (CONVERTER.EXCHANGE, "exch_ts")]:
                routed = events[(events["ev"] & mask) != 0]
                self.assertEqual(len(routed), 29)
                self.assertTrue(np.all(routed[field][1:] >= routed[field][:-1]))
            config = workspace / "replay.yaml"
            result_file = workspace / "result.json"
            config.write_text(yaml.safe_dump({
                "source": str(destination / "manifest.json"), "output": str(result_file),
                "date": "2025-02-01", "symbol": "BTCUSDT", "tick_size": 0.1, "lot_size": 0.001,
                "depth_levels": 10, "snapshot_interval_ns": 100_000_000, "latency_sample_every": 0,
            }))
            subprocess.run([str(PROBE), str(config)], check=True)
            result = json.loads(result_file.read_text())
            for name in ["normalized_rows", "depth_updates", "snapshot_levels", "clear_events", "trade_events"]:
                self.assertEqual(result[name], manifest[name])
            self.assertEqual(result["local_events"], 29)
            self.assertEqual(result["exchange_events"], 29)
            self.assertEqual(result["final_bids"], [[98, 3]])
            self.assertEqual(result["final_asks"], [[102, 4]])
            self.assertEqual(result["shards_released"], len(manifest["shards"]))

    # 【非核心】检查未知压缩格式与非正行情延迟会明确报错。
    def test_unknown_compression_and_nonpositive_latency_fail(self):
        with self.assertRaises(ValueError):
            CONVERTER.open_market_csv(Path("unknown.csv.zip"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = root / "book.csv.gz"
            trades = root / "trades.csv.gz"
            with gzip.open(book, "wt") as stream:
                stream.write("exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount\n"
                             "binance-futures,BTCUSDT,100,90,true,bid,99.9,1\n"
                             "binance-futures,BTCUSDT,100,90,true,ask,100.1,1\n")
            with gzip.open(trades, "wt") as stream:
                stream.write("exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
                             "binance-futures,BTCUSDT,110,120,1,buy,100.1,1\n")
            with self.assertRaisesRegex(ValueError, "不会自动平移"):
                CONVERTER.convert(book, trades, "2026-07-12", root / "output", full_day=False)
            self.assertFalse(list((root / "output").glob("*.npz")))

    # 【非核心】检查回放配置拒绝缺失项、重复键及未完整解析的数值。
    def test_replay_config_requires_all_fields_and_exact_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "source": "synthetic", "date": "synthetic", "symbol": "BTCUSDT",
                "tick_size": 0.1, "lot_size": 0.001, "depth_levels": 10,
                "snapshot_interval_ns": 100_000_000, "latency_sample_every": 0,
                "output": str(root / "report.json"),
            }
            missing = dict(config)
            del missing["tick_size"]
            malformed = {**config, "snapshot_interval_ns": "100ms"}
            negative = {**config, "depth_levels": -1}
            cases = [yaml.safe_dump(missing), yaml.safe_dump(malformed), yaml.safe_dump(negative),
                     yaml.safe_dump(config) + "tick_size: 1\n"]
            filename = root / "replay.yaml"
            for content in cases:
                filename.write_text(content)
                result = subprocess.run([str(PROBE), str(filename)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("config field", result.stderr)

    # 【非核心】检查非法价格网格与未覆盖完整交易日的源数据被拒绝。
    def test_data_boundaries_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            CONVERTER.validate_grid(100.05, 1, 0.1, 0.001)
        with self.assertRaises(ValueError):
            CONVERTER.validate_day([{
                "file": "truncated.csv.gz",
                "first_local_timestamp_ns": 0,
                "last_local_timestamp_ns": 1,
            }], "2025-02-01")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
