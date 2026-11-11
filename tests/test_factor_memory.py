"""内存有界的因子分块须与一次性历史计算保持一致。"""

import datetime as dt
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from mm_research.data.storage import write_parquet
from mm_research.execution import ordered_process_map
from mm_research.factors.dataset import compute_factor_chunk, factor_module


# 【非核心】返回进程编号，用于验证独立分块不复用上一个分配器。
def worker_identity(task: int) -> int:
    """每份任务只返回当前PID，不申请额外内存。"""
    import os
    return os.getpid()


class FactorMemoryTests(unittest.TestCase):
    """长时间不变、午夜、缺失和来源边界均保留原公式。"""

    # 【非核心】真实分块入口携带年龄及方向状态，比较全部36列与整段计算。
    def test_stateful_chunks_match_whole_history(self):
        """在两种日界上检查状态、80%窗口及首尾空值逐行相同。"""
        config = {"tick_size": .1, "min_fraction": .8, "aggregation_windows_s": [10, 60, 180, 600],
                  "mean_diff_windows_s": [[10, 60], [60, 180], [180, 600]], "write_threads": 1,
                  "output_interval_ns": 1_000_000_000}
        for last_date in ["2026-07-12", "2025-10-01"]:
            midnight = int(dt.datetime.fromisoformat(last_date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
            timestamps = midnight - 700_000_000_000 + np.arange(14000, dtype=np.int64) * 100_000_000
            dates = [str(dt.date.fromisoformat(last_date) - dt.timedelta(days=1)), last_date]
            source = pl.DataFrame({"symbol": ["BTCUSDT"] * 14000,
                                   "date": [dates[0]] * 7000 + [dates[1]] * 7000,
                                   "timestamp_ns": timestamps,
                                   "book_segment_id": [0] * 14000,
                                   "flow_valid": [True] * 14000,
                                   "bid_price_1": [100.] + [101.] * 13999,
                                   "ask_price_1": [103.] + [104.] * 13999,
                                   "mid_price": [101.5] + [102.5] * 13999,
                                   "bid_quantity_1": [1.] + [2.] * 13999,
                                   "ask_quantity_1": [3.] + [4.] * 13999,
                                   "trade_count": [1.] * 14000,
                                   "buy_volume": [1.] * 14000,
                                   "sell_volume": [0.] * 14000})
            # 第一次切块跨越一条超过600秒的连续方向；日界后插入真实缺失。
            source = source.with_columns([
                pl.when(pl.int_range(pl.len()).is_between(7750, 7765)).then(None).otherwise(pl.col(name)).alias(name)
                for name in ["buy_volume", "bid_price_1"]
            ])
            config["dates"] = dates
            for batch in ["gen014", "gen021"]:
                module = factor_module(batch)
                dependencies = module.source_expressions(config)
                formulas = module.factor_expressions(config)
                expected = source.select("symbol", "date", "timestamp_ns", *[
                    expression.alias(name) for name, expression in dependencies.items()
                ]).select("symbol", "date", "timestamp_ns", *[
                    expression.alias(name) for name, expression in formulas.items()
                ]).filter(pl.col("timestamp_ns") % 1_000_000_000 == 0)
                with tempfile.TemporaryDirectory() as temporary:
                    folder = Path(temporary)
                    dataset = {"dates": dates, "interval_ns": 100_000_000}
                    for name in source.columns[3:]:
                        write_parquet(source.select("symbol", "date", "timestamp_ns", name).lazy(),
                                      folder / f"{name}.parquet", {"kind": "feature", "dataset": dataset, "definition": {}})
                    carry = None
                    observed = []
                    for index, date in enumerate(dates):
                        usage = compute_factor_chunk({"config": config, "features": str(folder), "dates": [date],
                                                      "batch": batch, "names": list(formulas), "state": carry,
                                                      "staging": str(folder / "chunks"), "index": index,
                                                      "metadata": {name: {"kind": "factor", "dataset": dataset, "definition": {}} for name in formulas}})
                        carry = usage["state"]
                        table = None
                        for name in formulas:
                            values = pl.read_parquet(folder / f"chunks/{index:04d}/{name}.parquet")
                            table = values if table is None else table.with_columns(values[name])
                        observed.append(table)
                    assert_frame_equal(pl.concat(observed), expected, check_exact=False, abs_tol=1e-9)

    # 【非核心】一个worker连续两份任务也必须退出重建，释放已保留的分配器空间。
    def test_process_recycling(self):
        identities = list(ordered_process_map(worker_identity, [0, 1], 1, recycle_workers=True))
        self.assertNotEqual(identities[0], identities[1])


if __name__ == "__main__":
    unittest.main()
