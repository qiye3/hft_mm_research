"""基础行情文件适配，供研究包与已有离线转换工具共享压缩 CSV 读取。

open_market_csv 保留压缩读取语义，build_trade_bins 汇总成交，build_snapshot_grid 对齐快照。
build_snapshot_base 关联盘口与成交，并叠加已有质量参考。
输入为同源原始文件，输出为保留质量标记且不补造事件路径的 100ms 基础表。
"""

import datetime as dt
import gzip
from pathlib import Path

import numpy as np
import polars as pl
import zstandard

DAY_NS = 86_400_000_000_000


# ==================== 原始读取与时间网格 ====================
# 按源文件压缩格式建立文本流，供盘口与成交流共用。
def open_market_csv(filename: Path):
    """打开 gzip 或 Zstandard 压缩的行情 CSV。

    filename 为既有源文件，.zst/.zstd 使用 Zstandard，其余延续原有 gzip 读取方式。
    返回可用于 with 的文本流，换行交给 csv.reader 处理，关闭时同时释放底层文件。
    两种压缩格式共用后续字段、时间和盘口检查。
    """
    if filename.suffix in {".zst", ".zstd"}:
        return zstandard.open(filename, "rt", encoding="utf-8", newline="")
    return gzip.open(filename, "rt", newline="")


# 以 UTC 日期建立完整网格，时间戳统一保留为纳秒整数。
def make_day_grid(date: str, interval_ns: int) -> pl.DataFrame:
    """输入 UTC 日期与正的采样间隔，返回覆盖 [日初,次日日初) 的时间表。

    网格必须整除一天，标签和交易窗口使用相同的右端点时间。
    """
    if interval_ns <= 0 or DAY_NS % interval_ns:
        raise ValueError("采样间隔必须为一天的正整数约数")
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    return pl.DataFrame({"timestamp_ns": np.arange(start, start + DAY_NS, interval_ns, dtype=np.int64)})


# 核对源文件的时序、品种和交易日首尾覆盖。
def validate_source_frame(table: pl.DataFrame, clock: str, date: str) -> None:
    """输入已投影的原始行情及接收时钟列，非法品种、倒序和截断日直接报错。

    首尾一分钟覆盖只检查日界，盘中数据中断另由年龄和编号质量标记表达。
    """
    if table.is_empty() or not table[clock].is_sorted():
        raise ValueError(f"源文件为空或接收时间倒退 {date}")
    if table.filter((pl.col("exchange") != "binance-futures") | (pl.col("symbol") != "BTCUSDT")).height:
        raise ValueError("快照入口仅接收 binance-futures BTCUSDT")
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    first = table[clock][0]
    last = table[clock][-1]
    if not start <= first < start + 60_000_000_000 or not start + DAY_NS - 60_000_000_000 <= last < start + DAY_NS:
        raise ValueError(f"源文件未覆盖当日首尾一分钟 {date}")


# ==================== 成交流汇总与快照对齐 ====================
# 用 Polars 计算左开右闭成交窗口，同时记录因果可见的编号跳跃。
def build_trade_bins(filename: Path, date: str, interval_ns: int, lot_size: float) -> pl.DataFrame:
    """输入同源原始成交，返回以 timestamp_ns 为键的数量、金额、笔数与编号标记。

    先按数量步长转成整数，ceil(local_ts/interval) 确定右端点，避免空窗浮点残差。
    编号跳跃只标记当前成交收到后的网格，所有原始成交都参与数据质量检查。
    """
    table = pl.scan_csv(filename, schema_overrides={
        "timestamp": pl.Int64, "local_timestamp": pl.Int64, "id": pl.Int64,
        "price": pl.Float64, "amount": pl.Float64,
    }).select(
        "exchange", "symbol", "id", "side", "price", "amount",
        (pl.col("timestamp") * 1000).alias("trade_exchange_ns"),
        (pl.col("local_timestamp") * 1000).alias("trade_local_ns"),
    ).collect(engine="streaming")
    validate_source_frame(table, "trade_local_ns", date)
    invalid = (~pl.col("price").is_finite() | ~pl.col("amount").is_finite()
               | (pl.col("price") <= 0) | (pl.col("amount") <= 0)
               | ~pl.col("side").is_in(["buy", "sell"]))
    if table.filter(invalid).height or table["id"].n_unique() != table.height:
        raise ValueError(f"成交包含非法值或重复编号 {date}")
    lots = pl.col("amount") / lot_size
    if table.filter((lots - lots.round()).abs() > 1e-6).height:
        raise ValueError(f"成交数量偏离 Lot Size {date}")
    table = table.with_columns(
        (((pl.col("trade_local_ns") + interval_ns - 1) // interval_ns) * interval_ns).alias("timestamp_ns"),
        (pl.col("id").diff().fill_null(1) != 1).alias("trade_sequence_gap"),
        lots.round().cast(pl.Int64).alias("quantity_lots"),
    )
    return table.group_by("timestamp_ns", maintain_order=True).agg(
        pl.len().cast(pl.Int64).alias("trade_count"),
        (pl.col("quantity_lots").filter(pl.col("side") == "buy").sum() * lot_size).alias("buy_volume"),
        (pl.col("quantity_lots").filter(pl.col("side") == "sell").sum() * lot_size).alias("sell_volume"),
        (pl.col("price") * pl.col("amount")).filter(pl.col("side") == "buy").sum().alias("buy_notional"),
        (pl.col("price") * pl.col("amount")).filter(pl.col("side") == "sell").sum().alias("sell_notional"),
        pl.col("trade_sequence_gap").any(),
        pl.col("trade_local_ns").last().alias("last_trade_timestamp_ns"),
        pl.col("trade_exchange_ns").last().alias("last_trade_exchange_ns"),
    )


# 将快照按接收时钟向后对齐，供独立盘口研究与完整基础表共用。
def build_snapshot_grid(snapshot: Path, date: str, interval_ns: int,
                        max_snapshot_age_ns: int) -> pl.DataFrame:
    """输入 25 档快照、UTC 日期、纳秒网格和年龄阈值，返回前十档网格表。

    仅使用采样时刻已收到的快照，book_valid 标记完整、有序且未过期的盘口。
    输出保留 exchange 字段，完整基础表会在关联成交后移除该字段。
    """
    price_columns = []
    schema = {"timestamp": pl.Int64, "local_timestamp": pl.Int64}
    ordered_fields = []
    for level in range(10):
        for side, prefix in [("bids", "bid"), ("asks", "ask")]:
            for raw, output in [("price", "price"), ("amount", "quantity")]:
                name = f"{side}[{level}].{raw}"
                schema[name] = pl.Float64
                price_columns.append(pl.col(name).alias(f"{prefix}_{output}_{level + 1}"))
                ordered_fields.append(f"{prefix}_{output}_{level + 1}")
    snapshots = pl.scan_csv(snapshot, schema_overrides=schema).select(
        "exchange", "symbol", (pl.col("timestamp") * 1000).alias("book_exchange_ns"),
        (pl.col("local_timestamp") * 1000).alias("last_book_timestamp_ns"), *price_columns,
    ).collect(engine="streaming")
    validate_source_frame(snapshots, "last_book_timestamp_ns", date)
    snapshots = snapshots.unique(subset="last_book_timestamp_ns", keep="last", maintain_order=True)
    grid = make_day_grid(date, interval_ns)
    base = grid.join_asof(snapshots, left_on="timestamp_ns", right_on="last_book_timestamp_ns", strategy="backward")
    base = base.with_columns(
        (pl.col("timestamp_ns") - pl.col("last_book_timestamp_ns")).alias("snapshot_age_ns"),
        ((pl.col("bid_price_1") + pl.col("ask_price_1")) * 0.5).alias("mid_price"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread"),
        pl.lit(date).alias("date"), pl.lit("BTCUSDT").alias("symbol"),
    )
    valid_levels = [pl.col(field).is_finite() & (pl.col(field) > 0) for field in ordered_fields]
    for level in range(1, 10):
        valid_levels.extend([
            pl.col(f"bid_price_{level}") > pl.col(f"bid_price_{level + 1}"),
            pl.col(f"ask_price_{level}") < pl.col(f"ask_price_{level + 1}"),
        ])
    base = base.with_columns(
        (pl.all_horizontal(valid_levels) & (pl.col("spread") > 0)
         & (pl.col("snapshot_age_ns") <= max_snapshot_age_ns)).fill_null(False).alias("book_valid"),
    )
    return base


# 关联快照网格与同源成交，叠加已知质量区段并保留未知 OFI 为空。
def build_snapshot_base(snapshot: Path, trades: Path, date: str, interval_ns: int,
                         lot_size: float, max_snapshot_age_ns: int,
                         reference: Path | None = None) -> pl.DataFrame:
    """输入快照、成交、采样参数和可选 C++ 参考，返回完整 100ms 基础表。

    复用 build_snapshot_grid 的盘口对齐，按左开右闭窗口关联成交。
    已有参考覆盖时叠加已识别缺口与恢复区段，供现有数据生产与研究使用。
    """
    base = build_snapshot_grid(snapshot, date, interval_ns, max_snapshot_age_ns)
    base = base.join(build_trade_bins(trades, date, interval_ns, lot_size), on="timestamp_ns", how="left")
    base = base.with_columns(
        pl.col("trade_count").fill_null(0), pl.col("trade_sequence_gap").fill_null(False),
        pl.col("buy_volume", "sell_volume", "buy_notional", "sell_notional").fill_null(0.0),
        pl.col("last_trade_timestamp_ns", "last_trade_exchange_ns").forward_fill().fill_null(0),
        pl.col("last_book_timestamp_ns").fill_null(0),
    )
    if reference is not None and reference.exists():
        known = pl.read_parquet(reference, columns=["timestamp_ns", "book_valid", "flow_valid", "segment_id"])
        known = known.rename({name: "reference_" + name for name in ["book_valid", "flow_valid", "segment_id"]})
        base = base.join(known, on="timestamp_ns", how="left", validate="1:1")
        if base["reference_book_valid"].null_count():
            raise ValueError(f"已有质量参考未覆盖完整网格 {date}")
        base = base.with_columns((pl.col("book_valid") & pl.col("reference_book_valid").cast(pl.Boolean)).alias("book_valid"))
    else:
        base = base.with_columns(
            pl.lit(True).alias("reference_flow_valid"), pl.lit(0).alias("reference_segment_id"),
        )
    previous_valid = pl.col("book_valid").shift(1).fill_null(False)
    return base.with_columns(
        (pl.col("book_valid") & previous_valid & pl.col("reference_flow_valid").cast(pl.Boolean)).alias("flow_valid"),
        ((pl.col("book_valid") & ~previous_valid).cast(pl.Int64).cum_sum() + pl.col("reference_segment_id")).alias("segment_id"),
        pl.max_horizontal("last_book_timestamp_ns", "last_trade_timestamp_ns").alias("last_local_timestamp_ns"),
        pl.when(pl.col("last_trade_timestamp_ns") > pl.col("last_book_timestamp_ns"))
        .then(pl.col("last_trade_exchange_ns")).otherwise(pl.col("book_exchange_ns")).fill_null(0).alias("last_exchange_timestamp_ns"),
        pl.lit(0, dtype=pl.Int64).alias("segment_start_ns"),
        pl.lit(None, dtype=pl.Float64).alias("ofi_bbo"),
        pl.lit(None, dtype=pl.Int64).alias("event_count"),
    ).drop("exchange")
