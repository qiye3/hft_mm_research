"""按来源实际档数给快照附加逐记录质量标签，供标准盘口Parquet生产使用。

annotate_snapshot_records 输入统一字段的 LazyFrame，输出保持原顺序与原值的 LazyFrame。
盘口长静默使用疑点标签，已知缺口仅依据显式提供的核实区间标注。
"""

import polars as pl

from mm_research.data.quality import build_record_flags


# 【核心逻辑6】检查来源全部快照档位及采集间隔，把异常与缺口证据写入质量标签。
def annotate_snapshot_records(table: pl.LazyFrame, silence_ns: int,
                              confirmed_gaps: list[list[int]], depth: int) -> pl.LazyFrame:
    """输入源顺序快照、一秒等静默门限及已核实区间，返回带 quality_flags 的快照。

    区间使用该采集源的接收纳秒时钟，缺口标签放在其后的恢复记录。
    所有输入行与来源实际档数的原值保持，静默本身只标记为suspected。
    """
    prices = [pl.col(f"{side}_price_{level}") for level in range(1, depth + 1) for side in ["bid", "ask"]]
    quantities = [pl.col(f"{side}_quantity_{level}") for level in range(1, depth + 1) for side in ["bid", "ask"]]
    unordered = [] # 未排序的盘口

    for level in range(1, depth):
        unordered.extend([
            pl.col(f"bid_price_{level}") <= pl.col(f"bid_price_{level + 1}"), # 买单价格递减
            pl.col(f"ask_price_{level}") >= pl.col(f"ask_price_{level + 1}"), # 卖单价格递增
        ])
    previous = pl.col("received_timestamp_ns").shift(1) # 前一行的接收时间
    confirmed = pl.lit(False)
    for start, end in confirmed_gaps:
        confirmed = confirmed | ((previous <= start) & (pl.col("received_timestamp_ns") >= end))

    rules = {
        "missing_exchange_time": pl.col("exchange_timestamp_ns").is_null(),
        "source_received_time_unknown": pl.col("received_timestamp_ns").is_null(),
        "invalid_price": pl.any_horizontal([value.is_null() | ~value.is_finite() | (value <= 0) for value in prices]),
        "incomplete_depth": pl.any_horizontal([value.is_null() | ~value.is_finite() | (value <= 0) for value in quantities]),
        "crossed_book": pl.col("bid_price_1") >= pl.col("ask_price_1"),
        "unordered_book": pl.any_horizontal(unordered),
        "source_received_time_backwards": pl.col("received_timestamp_ns").diff() < 0,
        "gap_before_confirmed": confirmed,
        "gap_before_suspected": (pl.col("received_timestamp_ns") - previous > silence_ns) & ~confirmed.fill_null(False),
        "unexpected_market": pl.col("exchange").is_null() | pl.col("symbol").is_null()
                             | (pl.col("exchange") != "binance-futures") | (pl.col("symbol") != "BTCUSDT"),
    }

    return table.with_columns(build_record_flags(rules).alias("quality_flags"))
