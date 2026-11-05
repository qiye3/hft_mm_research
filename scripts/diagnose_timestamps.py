"""比较同日同一事件在原始 CSV 与 HFT Parquet 中的两套时间戳。

run_timestamp_diagnosis 对成交按编号匹配，对盘口按时间、方向和价格数量唯一匹配。
输出逐小时延迟分布及证据结论，不覆盖任何行情时间戳，也不自动估计校正量。
"""

import argparse
import csv
import io
import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import zstandard as zstd


# 先匹配成交，再对整日分布的盘口小窗口核对接收时差。
def run_timestamp_diagnosis(directory: Path, output: Path) -> None:
    """输入包含同日两种来源的目录，输出可重现的时间差诊断 JSON。

    成交保留全部编号，盘口每小时取前五秒，降低整日匹配的内存需求。
    匹配排除重复键，防止同一价格数量多次出现导致错误配对。
    """
    output.mkdir(parents=True, exist_ok=True)
    raw_trades = load_raw_time_samples(directory / "trades.csv.zst", "trades")
    hft_trades = pl.read_parquet(directory / "trades.parquet").select(
        pl.col("trade_id").cast(pl.Int64),
        (pl.col("event_time") * 1_000_000).alias("hft_exchange_ns"),
        (pl.col("trade_time") * 1_000_000).alias("hft_transaction_ns"),
        pl.col("received_time").alias("hft_local_ns"),
        (pl.col("price").cast(pl.Float64) * 10).round().cast(pl.Int64).alias("price_tick"),
        (pl.col("quantity").cast(pl.Float64) * 1000).round().cast(pl.Int64).alias("quantity_lot"),
        pl.when(pl.col("is_buyer_maker")).then(pl.lit("sell")).otherwise(pl.lit("buy")).alias("side"),
    )
    trades = summarize_matches(raw_trades, hft_trades, ["trade_id", "side", "price_tick", "quantity_lot"])
    coverage = summarize_trade_coverage(raw_trades, hft_trades)
    (output / "trade_coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n")
    raw_book = load_raw_time_samples(directory / "book.csv.zst", "book")
    batches = []
    columns = ["event_time", "transaction_time", "received_time", "event_type", "side", "price", "quantity"]
    for batch in pq.ParquetFile(directory / "orderbook.parquet").iter_batches(batch_size=250000, columns=columns):
        frame = pl.from_arrow(batch).filter(
            (pl.col("event_type") == "update") & ((pl.col("event_time") // 1000) % 3600 < 5)
        )
        batches.append(frame.select(
            (pl.col("event_time") * 1_000_000).alias("exchange_key"),
            (pl.col("event_time") * 1_000_000).alias("hft_exchange_ns"),
            (pl.col("transaction_time") * 1_000_000).alias("hft_transaction_ns"),
            pl.col("received_time").alias("hft_local_ns"), pl.col("side"),
            (pl.col("price").cast(pl.Float64) * 10).round().cast(pl.Int64).alias("price_tick"),
            (pl.col("quantity").cast(pl.Float64) * 1000).round().cast(pl.Int64).alias("quantity_lot"),
        ))
    hft_book = pl.concat(batches)
    book = summarize_matches(raw_book, hft_book, ["exchange_key", "side", "price_tick", "quantity_lot"])
    report = {
        "date": directory.name.removeprefix("baidu_"),
        "book_sampling": "每小时交易所时间的前五秒，排除快照与重复键",
        "trades": trades, "book": book,
        "decision": "保留两套原始接收时间；正式研究成套使用原始 CSV 来源，不做固定平移。",
        "attribution": "事件级匹配可确认两套接收时标不同。时钟校准日志、采集位置与生成代码仍需来源方提供，才能区分时钟偏移与真实链路延迟。",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "date": report["date"], "trades": trades["summary"], "book": book["summary"],
        "trade_coverage": coverage,
    }, ensure_ascii=False), flush=True)


# 流式读取 CSV，只保存时间诊断所需的字段和确定性窗口。
def load_raw_time_samples(filename: Path, kind: str) -> pl.DataFrame:
    """输入压缩行情及类型，返回成交全量或盘口每小时前五秒的紧凑表。

    时间单位从微秒转换为纳秒，价格和数量转成研究网格整数进行精确匹配。
    盘口的快照行排除，原始记录保持原样。
    """
    rows = []
    with filename.open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as stream:
            reader = csv.reader(io.TextIOWrapper(stream))
            columns = {name: index for index, name in enumerate(next(reader))}
            for row in reader:
                exchange_us = int(row[columns["timestamp"]])
                if kind == "book" and (exchange_us // 1_000_000 % 3600 >= 5 or row[columns["is_snapshot"]] == "true"):
                    continue
                key = int(row[columns["id"]]) if kind == "trades" else exchange_us * 1000
                rows.append((
                    key, exchange_us * 1000, int(row[columns["local_timestamp"]]) * 1000,
                    row[columns["side"]], round(float(row[columns["price"]]) * 10),
                    round(float(row[columns["amount"]]) * 1000),
                ))
    key_name = "trade_id" if kind == "trades" else "exchange_key"
    return pl.DataFrame(rows, schema=[
        (key_name, pl.Int64), ("raw_exchange_ns", pl.Int64), ("raw_local_ns", pl.Int64),
        ("side", pl.String), ("price_tick", pl.Int64), ("quantity_lot", pl.Int64),
    ], orient="row")


# 仅在两边都唯一的事件键上比较时钟，按小时检验差异的漂移。
def summarize_matches(raw: pl.DataFrame, hft: pl.DataFrame, keys: list[str]) -> dict:
    """输入两套紧凑事件表和匹配键，输出匹配率、重复量和三种延迟分布。

    delta 为 HFT 接收时间减原始接收时间。交易所时间差单列，防止混淆 E/T 字段。
    """
    raw_unique = raw.filter(~pl.struct(keys).is_duplicated())
    hft_unique = hft.filter(~pl.struct(keys).is_duplicated())
    matched = raw_unique.join(hft_unique, on=keys, how="inner", validate="1:1").with_columns(
        ((pl.col("hft_local_ns") - pl.col("raw_local_ns")) / 1e6).alias("delta_ms"),
        ((pl.col("raw_local_ns") - pl.col("raw_exchange_ns")) / 1e6).alias("raw_feed_ms"),
        ((pl.col("hft_local_ns") - pl.col("hft_exchange_ns")) / 1e6).alias("hft_feed_ms"),
        ((pl.col("hft_exchange_ns") - pl.col("raw_exchange_ns")) / 1e6).alias("exchange_difference_ms"),
        ((pl.col("hft_transaction_ns") - pl.col("raw_exchange_ns")) / 1e6).alias("transaction_difference_ms"),
        ((pl.col("raw_exchange_ns") // 3_600_000_000_000) % 24).alias("utc_hour"),
    )
    statistics = []
    for column in ["delta_ms", "raw_feed_ms", "hft_feed_ms", "exchange_difference_ms", "transaction_difference_ms"]:
        for quantile in [0.0, 0.01, 0.5, 0.99, 1.0]:
            statistics.append(pl.col(column).quantile(quantile).alias(f"{column}_p{quantile * 100:g}"))
    return {
        "raw_rows": raw.height, "hft_rows": hft.height,
        "raw_ambiguous_rows": raw.height - raw_unique.height,
        "hft_ambiguous_rows": hft.height - hft_unique.height,
        "matched_rows": matched.height,
        "summary": matched.select(statistics).to_dicts()[0],
        "by_hour": matched.group_by("utc_hour").agg(pl.len().alias("matches"), *statistics)
        .sort("utc_hour").to_dicts(),
    }


# 区分两套采集流的覆盖差异与同一编号下的内容冲突。
def summarize_trade_coverage(raw: pl.DataFrame, hft: pl.DataFrame) -> dict:
    """输入两套成交表，返回互相缺失的编号数量、小时分布和内容冲突数。

    HFT 的零价格或零数量记录单独统计，不将它们补成正常成交。
    这里描述两套观测的差异，无法仅凭差异决定哪套来源包含全部真实成交。
    """
    missing_hft = raw.join(hft.select("trade_id"), on="trade_id", how="anti")
    missing_raw = hft.join(raw.select("trade_id"), on="trade_id", how="anti")
    common = raw.join(hft, on="trade_id", how="inner", validate="1:1", suffix="_hft")
    wrong = common.filter(
        (pl.col("price_tick") != pl.col("price_tick_hft"))
        | (pl.col("quantity_lot") != pl.col("quantity_lot_hft"))
        | (pl.col("side") != pl.col("side_hft"))
    )
    hourly = {}
    for name, table, clock in [
        ("raw_only", missing_hft, "raw_exchange_ns"),
        ("hft_only_nonzero", missing_raw.filter((pl.col("price_tick") > 0) & (pl.col("quantity_lot") > 0)), "hft_transaction_ns"),
    ]:
        counts = table.with_columns(((pl.col(clock) // 3_600_000_000_000) % 24).alias("utc_hour"))
        hourly[name] = counts.group_by("utc_hour").len().sort("utc_hour").to_dicts()
    return {
        "raw_ids_missing_in_hft": missing_hft.height,
        "hft_ids_missing_in_raw": missing_raw.height,
        "hft_only_zero_records": missing_raw.filter((pl.col("price_tick") == 0) | (pl.col("quantity_lot") == 0)).height,
        "common_ids": common.height, "common_value_mismatches": wrong.height,
        "hourly": hourly,
    }


# 选择同日文件目录和诊断结果目录。
def main() -> None:
    """输入两个命令行目录，执行只读时戳诊断并保存结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    run_timestamp_diagnosis(arguments.directory, arguments.output)


if __name__ == "__main__":
    main()
