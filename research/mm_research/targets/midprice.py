"""从完整100ms研究时钟生成可跨日的精确未来Mid标签。"""

import polars as pl


# 【非核心】把纳秒预测周期转换为字段和目录中的时间后缀。
def horizon_name(horizon_ns: int) -> str:
    """只接受正整数毫秒，秒级研究周期使用1s、10s等后缀。"""
    if horizon_ns <= 0 or horizon_ns % 1_000_000:
        raise ValueError("预测周期必须是正整数毫秒")
    if horizon_ns % 1_000_000_000 == 0:
        return f"{horizon_ns // 1_000_000_000}s"
    return f"{horizon_ns // 1_000_000}ms"


# 【核心逻辑36】把逐日区段转换为可跨午夜的连续性标记，复用五周期质量前缀。
def prepare_label_context(base: pl.DataFrame, interval_ns: int, gap_markers: list[int],
                          source_markers: list[int]) -> pl.DataFrame:
    """日内使用已有区段变化，午夜由标准行情接续标记决定；所有质量检查保留100ms分辨率。"""
    same_day = pl.col("date") == pl.col("date").shift(1)
    segment_change = (pl.col("book_segment_id") != pl.col("book_segment_id").shift(1)) & same_day
    time_gap = pl.col("timestamp_ns").diff() != interval_ns
    valid_mid = pl.col("mid_price").is_finite() & (pl.col("mid_price") > 0)
    bad = ~(pl.col("book_valid") & valid_mid).fill_null(False)
    table = base.with_columns(
        bad.cast(pl.Int64).alias("_bad"),
        (segment_change.fill_null(False) | time_gap.fill_null(False)
         | pl.col("timestamp_ns").is_in(gap_markers)).cast(pl.Int64).alias("_gap"),
        pl.col("timestamp_ns").is_in(source_markers).cast(pl.Int64).alias("_source"),
    )
    return table.with_columns(
        pl.col("_bad").cum_sum().alias("_bad_prefix"),
        pl.col("_gap").cum_sum().alias("_gap_prefix"),
        pl.col("_source").cum_sum().alias("_source_prefix"),
    )


# 【核心逻辑37】从共享前缀核对整个预测区间，构造三单位标签及独立失效原因。
def build_midprice_labels(base: pl.DataFrame, tick_size: float, interval_ns: int,
                          horizon_ns: int) -> pl.DataFrame:
    """未来必须精确为t+h，质量覆盖[t,t+h]，缺口及来源切换按(t,t+h]计入。"""
    if interval_ns <= 0 or horizon_ns % interval_ns or tick_size <= 0:
        raise ValueError("周期必须整除基础网格，Tick Size须为正")
    steps = horizon_ns // interval_ns
    suffix = horizon_name(horizon_ns)
    future_mid = pl.col("mid_price").shift(-steps)
    future_timestamp = pl.col("timestamp_ns").shift(-steps)
    available = (future_timestamp == pl.col("timestamp_ns") + horizon_ns).fill_null(False)
    invalid_book = (pl.col("_bad_prefix").shift(-steps) - pl.col("_bad_prefix") + pl.col("_bad") > 0).fill_null(False)
    invalid_gap = (pl.col("_gap_prefix").shift(-steps) > pl.col("_gap_prefix")).fill_null(False)
    invalid_source = (pl.col("_source_prefix").shift(-steps) > pl.col("_source_prefix")).fill_null(False)
    valid = available & ~invalid_book & ~invalid_gap & ~invalid_source
    difference = future_mid - pl.col("mid_price")
    return base.select(
        "symbol", "date", "timestamp_ns", valid.alias("label_valid"),
        pl.when(valid).then((future_mid / pl.col("mid_price") - 1) * 10000).otherwise(None).alias(f"y_mid_{suffix}_bps"),
        pl.when(valid).then(difference / tick_size).otherwise(None).alias(f"y_mid_{suffix}_ticks"),
        pl.when(valid).then(difference).otherwise(None).alias(f"y_mid_{suffix}_usdt"),
        pl.when(valid).then(future_timestamp).otherwise(None).alias("label_end_timestamp_ns"),
        (~available).alias("invalid_future"), invalid_book.alias("invalid_book"),
        invalid_gap.alias("invalid_gap"), invalid_source.alias("invalid_source"),
    )
