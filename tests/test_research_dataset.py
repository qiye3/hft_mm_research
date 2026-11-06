"""验证 M2-A 的真实 C++ 回放接入、OFI、固定采样边界和 Python 标签。

使用小型确定性 NPZ 夹具检查同时间戳跨分片、深档晋升、快照恢复和缺口。
另用独立数组公式检查 Polars 特征与标签，以及追加未来数据对既有特征的影响。
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mm_research.features.book import BOOK_FEATURE_COLUMNS
from mm_research.features.order_flow import add_order_flow_features
from mm_research.features.transform import build_feature_group
from mm_research.targets.midprice import build_midprice_labels
from convert_tardis import EVENT_DTYPE, LOCAL, EXCHANGE, BUY, SELL, DEPTH, SNAPSHOT, CLEAR, TRADE

EXTRACTOR = Path(sys.argv[1]).resolve()


class ResearchDatasetTests(unittest.TestCase):
    """针对数据时间口径与计算公式执行小规模回归。"""

    # 【非核心】通过原生 NPZ 读取器验证事件恰落在网格和跨分片时的完整状态。
    def test_cpp_grid_ofi_trades_and_snapshot(self):
        """生成可手算的双时钟行情，检查 C++ 输出的 OFI、成交与快照有效标记。"""
        both = LOCAL | EXCHANGE
        events = [(both | CLEAR, 1_000_000, 10_000_000, 0, 0, 0, 0, 0)]
        for level in range(11):
            events.extend([
                (both | SNAPSHOT | BUY, 1_000_000, 10_000_000, 99.9 - level * 0.1, 1, 0, 0, 0),
                (both | SNAPSHOT | SELL, 1_000_000, 10_000_000, 100.1 + level * 0.1, 2, 0, 0, 0),
            ])
        events.extend([
            (both | DEPTH | BUY, 90_000_000, 100_000_000, 99.9, 3, 0, 0, 0),
            (both | TRADE | BUY, 90_000_000, 100_000_000, 100.1, 0.5, 0, 0, 0),
            (both | TRADE | SELL, 90_000_000, 100_000_000, 99.9, 0.2, 0, 0, 0),
            (both | DEPTH | BUY, 190_000_000, 200_000_000, 99.9, 0, 0, 0, 0),
            (both | DEPTH | SELL, 290_000_000, 300_000_000, 100.1, 1, 0, 0, 0),
            (both | CLEAR, 390_000_000, 400_000_000, 0, 0, 0, 0, 0),
        ])
        for level in range(10):
            events.extend([
                (both | SNAPSHOT | BUY, 390_000_000, 400_000_000, 98 - level * 0.1, 4, 0, 0, 0),
                (both | SNAPSHOT | SELL, 390_000_000, 400_000_000, 102 + level * 0.1, 5, 0, 0, 0),
            ])
        events.append((both | DEPTH | BUY, 590_000_000, 600_000_000, 98, 4, 0, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            # 同一时刻多条记录跨越分片，检查样本没有提前看到半批状态。
            shards = []
            for offset in range(0, len(events), 7):
                filename = f"events-{offset}.npz"
                np.savez_compressed(work / filename, data=np.array(events[offset:offset + 7], dtype=EVENT_DTYPE))
                shards.append({"file": filename})
            (work / "manifest.json").write_text(json.dumps({"shards": shards}))
            (work / "replay.yaml").write_text(
                f"source: {work / 'manifest.json'}\noutput: {work / 'unused.json'}\n"
                "snapshot_interval_ns: 100000000\n"
                "date: synthetic\nsymbol: BTCUSDT\ntick_size: 0.1\nlot_size: 0.001\n"
                "depth_levels: 10\nlatency_sample_every: 0\n"
            )
            command = [
                str(EXTRACTOR), str(work / "replay.yaml"), "0", "1800000000", "1000000000", str(work / "samples.csv"),
            ]
            subprocess.run(command, check=True, capture_output=True)
            original_bytes = (work / "samples.csv").read_bytes()
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(original_bytes, (work / "samples.csv").read_bytes())
            table = pl.read_csv(work / "samples.csv", null_values=["nan", "-nan", "inf", "-inf"])
            self.assertEqual(table.height, 18)
            self.assertEqual(table["book_valid"][0], 0)
            first = table.row(1, named=True)
            self.assertAlmostEqual(first["ofi_bbo"], 2)
            self.assertAlmostEqual(first["buy_volume"], 0.5)
            self.assertAlmostEqual(first["sell_volume"], 0.2)
            self.assertEqual(first["trade_count"], 2)
            self.assertEqual(first["bid_quantity_1"], 3)
            self.assertAlmostEqual(table["ofi_bbo"][2], -3)
            self.assertAlmostEqual(table["bid_price_10"][2], 98.9)
            self.assertAlmostEqual(table["ofi_bbo"][3], 1)
            self.assertEqual(table["flow_valid"][4], 0)
            self.assertEqual(table["bid_quantity_10"][4], 4)
            self.assertEqual(table["ask_quantity_10"][4], 5)
            self.assertEqual(table["book_valid"][-1], 0)

    # 【非核心】用线性变化行情独立核对滚动统计和未来价格差，并验证前缀不依赖未来。
    def test_python_features_labels_and_no_future_leakage(self):
        """输入人工网格表，检查 1s/5s 窗口、缺口隔离、日尾标签与前缀一致性。"""
        count = 140
        values = {
            "timestamp_ns": np.arange(count, dtype=np.int64) * 100_000_000,
            "symbol": ["BTCUSDT"] * count, "date": ["1970-01-01"] * count,
            "trade_segment_id": [1] * count,
            "segment_id": [1] * 80 + [2] * 60,
            "segment_start_ns": [0] * 80 + [8_000_000_000] * 60,
            "book_valid": [1] * count, "flow_valid": [1] * count,
            "book_segment_id": [1] * 80 + [2] * 60,
            "mid_price": 100 + np.arange(count) * 0.01,
            "spread": [0.2] * count, "ofi_bbo": [1.0] * count,
            "buy_volume": [2.0] * count, "sell_volume": [1.0] * count,
        }
        for level in range(1, 11):
            values[f"bid_price_{level}"] = [99.9 - level * 0.1] * count
            values[f"ask_price_{level}"] = [100.1 + level * 0.1] * count
            values[f"bid_quantity_{level}"] = [2.0] * count
            values[f"ask_quantity_{level}"] = [1.0] * count
        base = pl.DataFrame(values)
        config = {"tick_size": 0.1, "lot_size": 0.001}
        result = build_feature_group("book", base, config)
        order_flow = add_order_flow_features(base)
        returns = build_feature_group("returns", base, config)
        self.assertAlmostEqual(order_flow["ofi_1s"][60], 10)
        self.assertAlmostEqual(order_flow["ofi_5s"][60], 50)
        self.assertAlmostEqual(result["imbalance_10"][60], 1 / 3)
        labels = build_midprice_labels(base, 0.1, 100_000_000)
        self.assertAlmostEqual(labels["y_mid_1s_usdt"][60], 0.1)
        self.assertFalse(labels["label_valid"][75])
        self.assertFalse(returns["return_feature_valid"][85])
        self.assertTrue(returns["return_feature_valid"][135])
        self.assertFalse(labels.tail(10)["label_valid"].any())
        prefix = build_feature_group("book", base.head(70), config)
        assert_frame_equal(prefix.select(BOOK_FEATURE_COLUMNS), result.head(70).select(BOOK_FEATURE_COLUMNS), check_exact=True)
        broken = base.with_columns(pl.when(pl.col("timestamp_ns") == 6_500_000_000)
                                   .then(0).otherwise(pl.col("flow_valid")).alias("flow_valid"))
        broken_result = build_feature_group("trades", broken, config)
        self.assertTrue(build_midprice_labels(broken, 0.1, 100_000_000)["label_valid"][60])
        self.assertFalse(broken_result["trade_feature_valid"][66])
        empty_window = base.with_columns(
            pl.Series("buy_volume", [100.123] + [0.0] * (count - 1)),
            pl.lit(0.0).alias("sell_volume"),
        )
        empty_result = build_feature_group("trades", empty_window, config)
        self.assertEqual(empty_result["trade_imbalance_1s"][60], 0)
        self.assertTrue(empty_result["trade_feature_valid"][60])



if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
