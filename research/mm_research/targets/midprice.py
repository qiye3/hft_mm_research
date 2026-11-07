"""从本地可见中间价独立生成多预测周期标签，主单位为 bps。

build_midprice_labels 输出 bps、Ticks、USDT 三种等价目标和独立有效标志。
标签只读取基础行情与质量区段，特征定义或预热长度的变化不会改变标签工件。
"""

import polars as pl

DAY_NS = 86_400_000_000_000


# 【非核心】把预测周期转换为100ms、1s等字段和目录后缀。
def horizon_name(horizon_ns: int) -> str:
    """输入正的整毫秒周期，返回 100ms、500ms、1s 等标签字段后缀。"""
    if horizon_ns <= 0 or horizon_ns % 1_000_000:
        raise ValueError("预测周期必须是正的整毫秒")
    if horizon_ns % 1_000_000_000 == 0:
        return f"{horizon_ns // 1_000_000_000}s"
    return f"{horizon_ns // 1_000_000}ms"


# 【核心逻辑24】按精确未来Mid构造bps、Ticks和USDT标签，隔离跨日与盘口中断。
def build_midprice_labels(base: pl.DataFrame, tick_size: float, interval_ns: int,
                          horizon_ns: int = 1_000_000_000) -> pl.DataFrame:
    """输入单日基础表、价格步长、采样间隔及预测周期，返回键、标签和有效标志。

    未来时刻必须恰好为 t+h，区间内盘口有效，且没有跨过质量区段。
    y_bps 为简单价格收益乘 10000，y_ticks 以当前 Tick Size 换算，y_usdt 为价格差。
    """
    suffix = horizon_name(horizon_ns)
    if interval_ns <= 0 or horizon_ns % interval_ns or tick_size <= 0:
        raise ValueError("预测周期必须是采样间隔的整数倍，Tick Size 须为正")
    table = base
    steps = horizon_ns // interval_ns
    future_mid = pl.col("mid_price").shift(-steps)
    future_timestamp = pl.col("timestamp_ns").shift(-steps)
    future_segment = pl.col("book_segment_id").shift(-steps)
    valid_mid = pl.col("mid_price").is_not_null() & pl.col("mid_price").is_finite() & (pl.col("mid_price") > 0)
    valid_points = (pl.col("book_valid").cast(pl.Boolean) & valid_mid).cast(pl.Int32)
    horizon_valid = valid_points.rolling_sum(steps + 1).shift(-steps)
    label_valid = (
        (future_timestamp == pl.col("timestamp_ns") + horizon_ns)
        & (future_timestamp // DAY_NS == pl.col("timestamp_ns") // DAY_NS)
        & (future_segment == pl.col("book_segment_id"))
        & (horizon_valid == steps + 1)
    ).fill_null(False)
    difference = future_mid - pl.col("mid_price")
    keys = [column for column in ["symbol", "date", "timestamp_ns"] if column in table.columns]
    return table.select(
        *keys,
        label_valid.alias("label_valid"),
        pl.when(label_valid).then((future_mid / pl.col("mid_price") - 1) * 10000)
        .otherwise(None).alias(f"y_mid_{suffix}_bps"),
        pl.when(label_valid).then(difference / tick_size).otherwise(None).alias(f"y_mid_{suffix}_ticks"),
        pl.when(label_valid).then(difference).otherwise(None).alias(f"y_mid_{suffix}_usdt"),
        pl.when(label_valid).then(future_timestamp).otherwise(None).alias("label_end_timestamp_ns"),
    )
