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

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("convert_tardis", ROOT / "tools/convert_tardis.py")
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)
PROBE = Path(sys.argv[1]).resolve()


class ConversionTests(unittest.TestCase):
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
            config.write_text(f"source: {destination / 'manifest.json'}\noutput: {result_file}\ndepth_levels: 10\n")
            subprocess.run([str(PROBE), str(config)], check=True)
            result = json.loads(result_file.read_text())
            for name in ["normalized_rows", "depth_updates", "snapshot_levels", "clear_events", "trade_events"]:
                self.assertEqual(result[name], manifest[name])
            self.assertEqual(result["local_events"], 29)
            self.assertEqual(result["exchange_events"], 29)
            self.assertEqual(result["final_bids"], [[98, 3]])
            self.assertEqual(result["final_asks"], [[102, 4]])
            self.assertEqual(result["shards_released"], len(manifest["shards"]))

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
