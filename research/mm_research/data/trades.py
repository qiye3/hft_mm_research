"""整理单一官方成交来源，供标准逐笔 Parquet 生产使用。

prepare_official_trades 合并重复编号、标记内容冲突与异常日期，并按交易所时间排序。
官方档案中的编号空缺保持原样，行情接收时刻由回放的固定延迟模型生成。
"""

import datetime as dt

import polars as pl

from mm_research.data.quality import build_record_flags
from mm_research.data.sources import DAY_NS, TRADE_COLUMNS, TRADE_KEYS, TRADE_VALUES


# 合并官方文件中的重复编号，保留异常记录并标明内容差异。
def prepare_official_trades(table: pl.DataFrame, date: str) -> pl.DataFrame:
    """输入统一字段的官方成交表和 UTC 日期，返回有序标准逐笔。

    同编号副本保留第一条，内容不同附加 content_conflict，空编号逐条保留。
    日期范围异常加标签，编号不连续保持原样，所有时间均为交易所成交时间。
    """
    known = table.filter(pl.col("trade_id").is_not_null())
    variants = known.group_by(TRADE_KEYS).agg(
        pl.struct(TRADE_VALUES).n_unique().alias("_content_variants"),
        pl.len().alias("_copies"),
    )
    chosen = known.unique(subset=TRADE_KEYS, keep="first", maintain_order=True).join(variants, on=TRADE_KEYS, how="left")
    flags = build_record_flags({
        "content_conflict": pl.col("_content_variants") > 1,
        "duplicate_collapsed": pl.col("_copies") > 1,
    })
    chosen = chosen.with_columns(
        pl.concat_list("quality_flags", flags).list.unique(maintain_order=True).alias("quality_flags"),
    ).select(TRADE_COLUMNS)
    table = pl.concat([chosen, table.filter(pl.col("trade_id").is_null())], how="vertical")
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    flags = build_record_flags({
        "outside_trade_day": ~pl.col("exchange_timestamp_ns").is_between(start, start + DAY_NS, closed="left"),
    })
    return table.with_columns(
        pl.concat_list("quality_flags", flags).list.unique(maintain_order=True).alias("quality_flags"),
    ).sort(["exchange_timestamp_ns", "trade_id", "source_row"], nulls_last=True, maintain_order=True)
