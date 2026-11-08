"""检查文件级共享与独占锁以及原子发布失败后的完整性。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

import polars as pl

from mm_research.data.storage import file_locks, scan_series, write_parquet


class StorageTests(unittest.TestCase):
    """使用真实文件系统验证单变量文件更新边界。"""

    # 【非核心】多列读取按时间键归并，确认缺失键对齐与流式执行计划。
    def test_series_join_streams_on_sorted_keys(self):
        """核对跨日期关联和筛选结果，防止全日期查询退回内存关联。"""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            dataset = {"dates": ["2026-07-12", "2026-07-13"]}
            first = pl.DataFrame({
                "symbol": ["BTCUSDT"] * 3,
                "date": ["2026-07-12", "2026-07-12", "2026-07-13"],
                "timestamp_ns": [1, 2, 3], "first": [10.0, 20.0, 30.0],
            })
            second = first.select("symbol", "date", "timestamp_ns").filter(
                pl.col("timestamp_ns") != 2,
            ).with_columns(pl.Series("second", [100.0, 300.0]))
            for name, table in [("first", first), ("second", second)]:
                metadata = {"kind": "feature", "name": name, "dataset": dataset, "definition": {}}
                write_parquet(table.lazy(), folder / f"{name}.parquet", metadata)
            query, _ = scan_series(folder, ["first", "second"])
            self.assertEqual(query.collect(engine="streaming")["second"].to_list(), [100.0, None, 300.0])
            plan = query.show_graph(plan_stage="physical", engine="streaming", raw_output=True)
            self.assertIn("merge-join", plan)
            self.assertNotIn("in-memory-join", plan)
            filtered, _ = scan_series(folder, ["first", "second"], ["2026-07-13"])
            self.assertEqual(filtered.collect(engine="streaming")["timestamp_ns"].to_list(), [3])

    # 【非核心】同文件写入互斥，不同文件以及同文件共享读取可以并行。
    def test_file_lock_granularity(self):
        """以事件控制工作线程，确认锁范围和稳定锁文件。"""
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "value.parquet"
            entered = Event()

            # 【非核心】在取得指定文件锁后通知主测试线程。
            def acquire(path, shared):
                """输入目标及共享标记，取得锁后设置事件。"""
                with file_locks([path], shared=shared):
                    entered.set()

            with ThreadPoolExecutor(max_workers=1) as executor:
                with file_locks([filename]):
                    future = executor.submit(acquire, filename.with_name("other.parquet"), False)
                    future.result(timeout=3)
                    entered.clear()
                    future = executor.submit(acquire, filename, True)
                    self.assertFalse(entered.wait(0.1))
                future.result(timeout=3)
                entered.clear()
                with file_locks([filename], shared=True):
                    executor.submit(acquire, filename, True).result(timeout=3)
                    self.assertTrue(entered.is_set())
            self.assertTrue((filename.parent / ".locks/value.parquet.lock").exists())

    # 【非核心】注入写盘错误时，正式Parquet仍保留原内容且临时文件被清理。
    def test_atomic_publication_failure(self):
        """先发布一个文件，再模拟第二次发布失败。"""
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "value.parquet"
            metadata = {"kind": "feature", "name": "value", "dataset": {}, "definition": {}}
            original = pl.DataFrame({"symbol": ["BTCUSDT"], "date": ["2026-07-12"],
                                     "timestamp_ns": [0], "value": [1.0]}).lazy()
            with file_locks([filename]):
                write_parquet(original, filename, metadata)
                content = filename.read_bytes()
                with patch.object(pl.LazyFrame, "collect_batches", side_effect=OSError("磁盘错误")):
                    with self.assertRaises(OSError):
                        write_parquet(original, filename, metadata)
                self.assertEqual(filename.read_bytes(), content)
            self.assertFalse(list(filename.parent.glob(".publishing-*")))


if __name__ == "__main__":
    unittest.main()
