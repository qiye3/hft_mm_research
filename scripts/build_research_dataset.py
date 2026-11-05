"""M2-A 数据生产入口，将已有 NPZ 回放成可重复读取的按日 Parquet。

build_dataset_day 连接 C++ 提取与 Polars 特征构造，build_research_table 定义因果窗口和标签。
输入为日期配置和已转换行情，输出为基础表、研究表、字段说明和生产清单。
本阶段仅保存基础微观结构特征，不训练模型或生成交易信号。
"""

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import time
from pathlib import Path

import polars as pl
import numpy as np
import yaml

from convert_tardis import convert, open_market_csv
from validate_research_dataset import QUALITY_POLICY, validate_research_day

DAY_NS = 86_400_000_000_000
FEATURE_COLUMNS = [
    "spread_ticks", "imbalance_1", "imbalance_5", "imbalance_10",
    "bid_depth_10", "ask_depth_10", "microprice_minus_mid",
    "ofi_100ms", "ofi_1s", "ofi_5s", "buy_volume_1s", "sell_volume_1s",
    "buy_volume_5s", "sell_volume_5s", "trade_imbalance_100ms", "trade_imbalance_1s",
    "return_100ms_bps", "return_1s_bps", "return_5s_bps",
]


# 复用已完成日期，全部质量校验通过后发布研究数据清单。
def run_dataset_production(config_path: Path, rebuild: bool = False) -> None:
    """输入项目配置及重建开关，输出日期分区和包含字段列表的总清单。

    各日期独立运行和恢复，只有完整写出的 Parquet 才进入完成清单。
    路径相对项目根目录，已有原始数据和 NPZ 不在此重复生成。
    """
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(config_path.read_text())
    output_root = root / config["output_root"]
    reports = []
    for date in config["dates"]:
        date = str(date)
        report = build_dataset_day(root, config, date, rebuild)
        validation_file = root / config["result_root"] / f"date={date}" / "validation.json"
        if validation_file.exists():
            validation = json.loads(validation_file.read_text())
            if not validation["passed"] or validation.get("quality_policy") != QUALITY_POLICY:
                raise ValueError(f"日期校验未通过或使用了不同质量规则 {date}")
            if validation["valid_labels"] != report["valid_label_rows"]:
                raise ValueError(f"校验报告与研究分区不匹配 {date}")
        else:
            validate_research_day(
                root / config["output_root"] / f"date={date}",
                root / "workspace/data/raw" / f"baidu_{date}" / "snapshot25.csv.zst",
                validation_file,
            )
        reports.append(report)
    manifest = {
        "schema_version": "m2_a_v1",
        "sampling_clock": "local_received_time",
        "interval_ns": config["interval_ns"],
        "label_horizon_ns": 1_000_000_000,
        "feature_columns": FEATURE_COLUMNS,
        "label_column": "target_mid_change_1s",
        "quality_policy": QUALITY_POLICY,
        "validated": True,
        "days": reports,
    }
    manifest_file = output_root / "manifest.json.part"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    manifest_file.replace(output_root / "manifest.json")


# 回放一个完整交易日，保留基础 Parquet 供后续 Python 新特征复用。
def build_dataset_day(root: Path, config: dict, date: str, rebuild: bool) -> dict:
    """输入项目目录、生产配置和 UTC 日期，返回该日数据量与质量统计。

    CSV 仅用于 C++ 到 Python 的暂存，Parquet 采用 ZSTD 3。
    原始样本和研究表分别原子替换，完成标记最后写入，失败日期可单独重跑。
    """
    output = root / config["output_root"] / f"date={date}"
    result = root / config["result_root"] / f"date={date}"
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    completion = output / "complete.json"
    if completion.exists() and not rebuild:
        previous = json.loads(completion.read_text())
        previous_settings = {key: value for key, value in previous["config"].items() if key != "dates"}
        current_settings = {key: value for key, value in config.items() if key != "dates"}
        if previous_settings != current_settings:
            raise ValueError(f"日期已有不同配置的结果，请使用 --rebuild {date}")
        if not all((output / f"{name}.parquet").is_file() for name in ["base", "research"]):
            raise ValueError(f"完成标记对应的文件缺失，请使用 --rebuild {date}")
        print(f"复用已完成日期 {date}", flush=True)
        return previous
    completion.unlink(missing_ok=True)
    (result / "validation.json").unlink(missing_ok=True)
    source = root / config["normalized_root"] / f"BTCUSDT_{date}" / "manifest.json"
    if not source.exists():
        raw = root / "workspace/data/raw" / f"baidu_{date}"
        staging = source.parent.with_name(source.parent.name + ".building")
        # .building 仅属于本入口的未完成转换，可从不可变原始文件重新生成。
        if staging.exists():
            shutil.rmtree(staging)
        convert(raw / "book.csv.zst", raw / "trades.csv.zst", date, staging, shard_rows=250000)
        staging.rename(source.parent)
    source_info = json.loads(source.read_text())
    if not source_info["full_day_validated"] or source_info["local_timestamp_offset_ns"] != 0:
        raise ValueError(f"正式研究要求完整日及未经平移的本地时间 {date}")
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    replay_config = {
        "source": str(source), "date": date, "symbol": "BTCUSDT",
        "tick_size": source_info["tick_size"], "lot_size": source_info["lot_size"],
        "depth_levels": 10, "snapshot_interval_ns": config["interval_ns"],
        "latency_sample_every": 0, "output": str(result / "replay.json"),
    }
    replay_file = result / "replay.yaml"
    replay_file.write_text(yaml.safe_dump(replay_config))
    csv_file = result / "samples.csv"
    started = time.monotonic()
    command = [
        str(root / config["extractor"]), str(replay_file), str(start), str(start + DAY_NS),
        str(config["max_book_age_ns"]), str(csv_file),
    ]
    subprocess.run(command, check=True)
    base = pl.read_csv(csv_file, infer_schema_length=10000, null_values=["nan", "-nan", "inf", "-inf"])
    base = base.with_columns(pl.col(pl.Float64).fill_nan(None))
    base = base.with_columns(pl.lit(date).alias("date"), pl.lit("BTCUSDT").alias("symbol"))
    expected_rows = DAY_NS // config["interval_ns"]
    expected_grid = pl.Series("timestamp_ns", range(start, start + DAY_NS, config["interval_ns"]), dtype=pl.Int64)
    if not base["timestamp_ns"].equals(expected_grid) or base.height != expected_rows:
        raise ValueError(f"采样网格缺失或重复 {date}")
    if base.filter(pl.col("last_local_timestamp_ns") > pl.col("timestamp_ns")).height:
        raise ValueError(f"样本使用了未来事件 {date}")
    raw_trades = root / "workspace/data/raw" / f"baidu_{date}" / "trades.csv.zst"
    base = annotate_trade_sequence_gaps(base, raw_trades)
    research = build_research_table(base, source_info["tick_size"], config["interval_ns"], source_info["lot_size"])
    for name, table in [("base", base), ("research", research)]:
        temporary = output / f"{name}.parquet.part"
        table.write_parquet(temporary, compression="zstd", compression_level=3, row_group_size=250000)
        temporary.replace(output / f"{name}.parquet")
    report = {
        "date": date, "rows": base.height, "source": str(source.relative_to(root)),
        "source_logical_events": source_info["logical_events"],
        "valid_book_rows": int(base["book_valid"].sum()),
        "valid_feature_rows": int(research["feature_valid"].sum()),
        "valid_label_rows": int(research["label_valid"].sum()),
        "trade_sequence_gap_rows": int(base["trade_sequence_gap"].sum()),
        "usable_rows": research.filter(pl.col("feature_valid") & pl.col("label_valid")).height,
        "parquet_bytes": sum((output / f"{name}.parquet").stat().st_size for name in ["base", "research"]),
        "production_seconds": time.monotonic() - started,
        "config": config,
    }
    temporary_completion = output / "complete.json.part"
    temporary_completion.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary_completion.replace(completion)
    csv_file.unlink()
    print(json.dumps({key: report[key] for key in ["date", "rows", "usable_rows", "production_seconds"]}), flush=True)
    return report


# 将成交编号跳跃对应的未知区间映射到固定网格，作为保守质量标志。
def annotate_trade_sequence_gaps(base: pl.DataFrame, trades: Path) -> pl.DataFrame:
    """输入 C++ 基础表和同源原始成交，返回增加 trade_sequence_gap 的表。

    编号不连续只能在当前成交接收时被发现，标记该时刻所属区间，不反向改写历史。
    标记不声称缺失了真实成交，也不补造事件，由研究窗口统一排除并重新预热。
    """
    suspect = np.zeros(base.height, dtype=bool)
    start = int(base["timestamp_ns"][0])
    previous_id = None
    with open_market_csv(trades) as source:
        rows = csv.reader(source)
        columns = {name: index for index, name in enumerate(next(rows))}
        for row in rows:
            trade_id = int(row[columns["id"]])
            local = int(row[columns["local_timestamp"]]) * 1000
            if previous_id is not None and trade_id != previous_id + 1:
                index = (local - start + 99_999_999) // 100_000_000
                if 0 <= index < base.height:
                    suspect[index] = True
            previous_id = trade_id
    return base.with_columns(pl.Series("trade_sequence_gap", suspect))


# 用固定网格上的历史区间构造基础特征，标签单独向未来取值。
def build_research_table(base: pl.DataFrame, tick_size: float, interval_ns: int, lot_size: float = 0.001) -> pl.DataFrame:
    """输入单日固定 100ms 基础表及价格、数量步长，返回含特征、标签和有效标志的表。

    滚动窗口左开右闭，只使用截至当前样本的流量。标签取同日同区段 t+1s 的 Mid 差。
    快照、过期盘口、预热及日尾没有足够未来数据的行保留，但相应标志为假。
    """
    if interval_ns != 100_000_000:
        raise ValueError("m2_a_v1 固定使用 100ms 网格")
    table = base.with_columns(
        pl.col("book_valid").cast(pl.Boolean),
        (pl.col("flow_valid").cast(pl.Boolean) & ~pl.col("trade_sequence_gap")).alias("flow_valid"),
        (pl.col("spread") / tick_size).alias("spread_ticks"),
        pl.col("ofi_bbo").alias("ofi_100ms"),
    )
    table = table.with_columns(
        pl.col("segment_id").alias("engine_segment_id"),
        (pl.col("segment_id") + pl.col("trade_sequence_gap").cast(pl.Int64).cum_sum()).alias("segment_id"),
    )
    table = table.with_columns(
        pl.when(pl.col("segment_id").diff().fill_null(1) != 0)
        .then(pl.col("timestamp_ns")).otherwise(None).forward_fill().alias("segment_start_ns"),
    )
    depth_expressions = []
    for levels in [1, 5, 10]:
        bid_depth = pl.sum_horizontal([pl.col(f"bid_quantity_{level}") for level in range(1, levels + 1)])
        ask_depth = pl.sum_horizontal([pl.col(f"ask_quantity_{level}") for level in range(1, levels + 1)])
        depth_expressions.extend([
            bid_depth.alias(f"bid_depth_{levels}"),
            ask_depth.alias(f"ask_depth_{levels}"),
            ((bid_depth - ask_depth) / (bid_depth + ask_depth)).alias(f"imbalance_{levels}"),
        ])
    table = table.with_columns(depth_expressions)
    table = table.with_columns(
        ((pl.col("ask_price_1") * pl.col("bid_quantity_1") + pl.col("bid_price_1") * pl.col("ask_quantity_1"))
         / (pl.col("bid_quantity_1") + pl.col("ask_quantity_1")) - pl.col("mid_price")).alias("microprice_minus_mid"),
        pl.col("flow_valid").cast(pl.Int32).rolling_sum(50, min_samples=50).over("segment_id").alias("valid_bins_5s"),
    )
    rolling_expressions = []
    for suffix, samples in [("1s", 10), ("5s", 50)]:
        for source in ["ofi_bbo", "buy_volume", "sell_volume"]:
            target = "ofi" if source == "ofi_bbo" else source
            aggregate = pl.col(source)
            if source != "ofi_bbo":
                # 源数量已通过步长检查，整数累计使空成交窗口严格为零。
                aggregate = (aggregate / lot_size).round().cast(pl.Int64)
            rolling = aggregate.rolling_sum(samples, min_samples=samples).over("segment_id")
            if source != "ofi_bbo":
                rolling = rolling * lot_size
            rolling_expressions.append(
                rolling.alias(f"{target}_{suffix}")
            )
    for suffix, samples in [("100ms", 1), ("1s", 10), ("5s", 50)]:
        previous = pl.col("mid_price").shift(samples).over("segment_id")
        rolling_expressions.append(((pl.col("mid_price") / previous - 1) * 10000).alias(f"return_{suffix}_bps"))
    table = table.with_columns(rolling_expressions)
    for suffix, buy, sell in [
        ("100ms", "buy_volume", "sell_volume"),
        ("1s", "buy_volume_1s", "sell_volume_1s"),
    ]:
        total = pl.col(buy) + pl.col(sell)
        table = table.with_columns(
            pl.when(total > 0).then((pl.col(buy) - pl.col(sell)) / total)
            .when(total == 0).then(0.0).otherwise(None).alias(f"trade_imbalance_{suffix}")
        )
    # 只在区间全程有效且同一恢复区段内生成标签，不能跨过快照或数据中断。
    future_mid = pl.col("mid_price").shift(-10)
    future_timestamp = pl.col("timestamp_ns").shift(-10)
    future_segment = pl.col("segment_id").shift(-10)
    horizon_valid_bins = (pl.col("book_valid") & pl.col("flow_valid")).cast(pl.Int32).rolling_sum(11).shift(-10)
    label_valid = (
        (future_timestamp == pl.col("timestamp_ns") + 1_000_000_000)
        & (future_segment == pl.col("segment_id")) & (horizon_valid_bins == 11)
    ).fill_null(False)
    table = table.with_columns(
        (pl.col("book_valid") & (pl.col("valid_bins_5s") == 50)
         & (pl.col("timestamp_ns") - pl.col("segment_start_ns") >= 5_000_000_000)
         & pl.all_horizontal([pl.col(column).is_not_null() & pl.col(column).is_finite()
                               for column in FEATURE_COLUMNS])).fill_null(False).alias("feature_valid"),
        label_valid.alias("label_valid"),
        pl.when(label_valid).then(future_mid - pl.col("mid_price")).otherwise(None).alias("target_mid_change_1s"),
        pl.when(label_valid).then((future_mid / pl.col("mid_price") - 1) * 10000)
        .otherwise(None).alias("target_mid_return_1s_bps"),
        pl.when(label_valid).then(future_timestamp).otherwise(None).alias("label_end_timestamp_ns"),
    )
    return table.drop("valid_bins_5s")


# 从命令行选择生产配置，允许显式重建已有日期。
def main() -> None:
    """输入配置路径和重建标志，执行按日数据生产并返回进程状态。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()
    run_dataset_production(arguments.config, arguments.rebuild)


if __name__ == "__main__":
    main()
