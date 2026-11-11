"""独立参考核对整秒跨日标签、300秒质量区间和来源边界。"""

import datetime as dt
import unittest

import numpy as np
import polars as pl

from mm_research.targets.midprice import build_midprice_labels, prepare_label_context
from mm_research.evaluation.dataset import partition_sample_mask


class TargetTests(unittest.TestCase):
    """两个UTC日的边界内含足够300s未来，分辨率保持100ms。"""

    # 【非核心】独立按数组索引核对跨午夜的五周期三单位标签。
    def test_cross_midnight_five_horizons(self):
        """逐日区段均从1开始，正常午夜接续后仍可得到未来300秒目标。"""
        midnight = int(dt.datetime(2026, 7, 26, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
        timestamps = midnight - 310_000_000_000 + np.arange(6201, dtype=np.int64) * 100_000_000
        dates = np.where(timestamps < midnight, "2026-07-25", "2026-07-26")
        mid = 60000 + np.sin(np.arange(len(timestamps)) / 100)
        base = pl.DataFrame({"symbol": ["BTCUSDT"] * len(mid), "date": dates, "timestamp_ns": timestamps,
                             "mid_price": mid, "book_valid": [True] * len(mid), "book_segment_id": np.where(timestamps < midnight, 7, 1)})
        context = prepare_label_context(base, 100_000_000, [], [])
        for seconds in [1, 10, 60, 180, 300]:
            result = build_midprice_labels(context, 0.1, 100_000_000, seconds * 1_000_000_000)
            steps = seconds * 10
            np.testing.assert_allclose(result[f"y_mid_{seconds}s_bps"].to_numpy()[:-steps], np.log(mid[steps:] / mid[:-steps]) * 10000, atol=1e-10)
            self.assertTrue(result["label_valid"][3000])
            self.assertFalse(result["label_valid"][-1])

    # 【非核心】预测区间内任一100ms无效点和已知来源切换均使标签置空。
    def test_path_quality_and_source_switch(self):
        """只检查秒末端会漏掉的100ms缺失点也必须被发现。"""
        timestamps = np.arange(5001, dtype=np.int64) * 100_000_000
        base = pl.DataFrame({"symbol": ["BTCUSDT"] * len(timestamps), "date": ["2026-07-25"] * len(timestamps),
                             "timestamp_ns": timestamps, "mid_price": [60000.0] * len(timestamps),
                             "book_valid": [True] * len(timestamps), "book_segment_id": [1] * len(timestamps)})
        damaged = base.with_columns(pl.when(pl.col("timestamp_ns") == 15_100_000_000).then(False).otherwise(pl.col("book_valid")).alias("book_valid"))
        context = prepare_label_context(damaged, 100_000_000, [30_000_000_000], [100_000_000_000])
        labels = build_midprice_labels(context, 0.1, 100_000_000, 10_000_000_000)
        self.assertTrue(labels["invalid_book"][100])
        self.assertTrue(labels["invalid_gap"][250])
        self.assertTrue(labels["invalid_source"][950])
        self.assertTrue(labels["label_valid"][1000])
        self.assertIsNone(labels["y_mid_10s_bps"][950])

    # 【非核心】训练末端300秒隔离和目标时刻检查独立生效。
    def test_training_boundary_purge(self):
        """跨日标签可存在，训练样本选择须排除越界与末端隔离区。"""
        boundary = int(dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
        table = pl.DataFrame({"timestamp_ns": [boundary-301_000_000_000, boundary-300_000_000_000, boundary-400_000_000_000],
                              "label_end_timestamp_ns": [boundary-1_000_000_000, boundary-1_000_000_000, boundary]})
        config = {"train_end": "2026-05-31", "validation_end": "2026-08-31", "purge_ns": 300_000_000_000}
        self.assertEqual(table.select(partition_sample_mask("2026-05-31", config)).to_series().to_list(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
