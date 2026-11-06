"""研究数据质量规则与独立参考校验。

本模块检查网格、盘口快照、成交分桶、编号疑点和未来标签，保存可复查的质量报告。
prepare_quality_state 统一特征和标签的因果区段，输入与输出均保留原始时间信息。
"""

import csv
import io
import json
from pathlib import Path

import numpy as np
import polars as pl
import zstandard as zstd

from mm_research.data.base import open_market_csv

QUALITY_POLICY = "causal_trade_id_gap_detection_and_5s_rewarm"


# 检查一个日期的标签、有效标记和独立盘口参考。
def validate_research_day(partition: Path, snapshot: Path, output: Path) -> dict:
    """输入日期分区、校验快照和报告位置，返回验收结果并在失败时抛错。

    对有效标签逐行重算未来价格差，盘口与独立 25 档数据向后对齐，不使用未来快照。
    """
    base = pl.read_parquet(partition / "base.parquet")
    research = pl.read_parquet(partition / "research.parquet")
    if not (base["timestamp_ns"].diff().drop_nulls() == 100_000_000).all():
        raise ValueError("研究网格不连续")
    if base.filter(pl.col("last_local_timestamp_ns") > pl.col("timestamp_ns")).height:
        raise ValueError("样本包含未来事件")
    label_rows = research.filter(pl.col("label_valid"))
    expected = research.select(
        "timestamp_ns", (pl.col("mid_price").shift(-10) - pl.col("mid_price")).alias("expected_label"),
    )
    checked = label_rows.join(expected, on="timestamp_ns", validate="1:1")
    if checked.filter((pl.col("target_mid_change_1s") - pl.col("expected_label")).abs() > 1e-10).height:
        raise ValueError("未来 1s 标签不匹配")
    if label_rows.filter(pl.col("label_end_timestamp_ns") != pl.col("timestamp_ns") + 1_000_000_000).height:
        raise ValueError("标签结束时刻不等于采样时刻加 1s")
    if research.tail(10)["label_valid"].any():
        raise ValueError("日末缺少未来时刻的标签应无效")
    reference = validate_snapshot_reference(base, snapshot)
    trades = validate_trade_bins(base, snapshot.with_name("trades.csv.zst"))
    report = {
        "date": partition.name.removeprefix("date="), "rows": base.height,
        "valid_labels": label_rows.height, "snapshot_reference": reference, "trade_reference": trades,
        "trade_sequence_gap_rows": int(base["trade_sequence_gap"].sum()),
        "quality_policy": QUALITY_POLICY,
        "book_gap_summary": summarize_book_gaps(base),
        "passed": reference["mismatched_samples"] == 0 and reference["compared_samples"] > 0
                  and reference["compared_samples"] == int(base["book_valid"].sum())
                  and reference["snapshot_times_backwards"] == 0
                  and trades["mismatched_flow_bins"] == 0 and trades["mismatched_count_bins"] == 0
                  and trades["duplicate_trade_ids"] == 0 and trades["nonpositive_trades"] == 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError(f"独立快照校验失败，详见 {output}")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


# 汇总被剔除的盘口时间段，保留中断长度供研究前检查。
def summarize_book_gaps(base: pl.DataFrame) -> dict:
    """输入固定网格基础表，返回无效盘口总量、最长未更新时长及连续区间。

    日初无盘口与盘中中断均保留在区间表，时间使用 UTC Unix 纳秒。
    最大年龄计算排除尚无盘口的零值，避免将日初误算成几十年的延迟。
    """
    invalid = base.filter(pl.col("book_valid") == 0)
    intervals = invalid.with_columns(
        (pl.col("timestamp_ns").diff().fill_null(100_000_000) != 100_000_000).cum_sum().alias("gap_id")
    ).group_by("gap_id").agg(
        pl.col("timestamp_ns").min().alias("start_timestamp_ns"),
        pl.col("timestamp_ns").max().alias("end_timestamp_ns"),
        pl.len().alias("rows"),
    ).sort("start_timestamp_ns").drop("gap_id")
    age = base.filter(pl.col("last_book_timestamp_ns") > 0).select(
        (pl.col("timestamp_ns") - pl.col("last_book_timestamp_ns")).max()
    ).item()
    return {"invalid_rows": invalid.height, "max_book_age_ns": age, "intervals": intervals.to_dicts()}


# 直接从原始成交计算网格归属，核对 C++ 的买卖量、金额与成交笔数。
def validate_trade_bins(base: pl.DataFrame, filename: Path) -> dict:
    """输入基础表和同源原始成交，返回独立分桶后的比较结果。

    用 ceil((local_ts-start)/100ms) 将事件归入左开右闭区间。
    快照或缺口导致的无效流量区间跳过金额比较，笔数仍逐区间核对。
    """
    expected = np.zeros((base.height, 4), dtype=np.float64)
    counts = np.zeros(base.height, dtype=np.int64)
    start = int(base["timestamp_ns"][0])
    seen_ids = set()
    duplicate_ids = 0
    nonpositive = 0
    id_gaps = 0
    missing_id_slots = 0
    previous_id = None
    previous_local = None
    maximum_gap_ns = 0
    with filename.open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as stream:
            reader = csv.reader(io.TextIOWrapper(stream))
            columns = {name: index for index, name in enumerate(next(reader))}
            for row in reader:
                local = int(row[columns["local_timestamp"]]) * 1000
                trade_id = int(row[columns["id"]])
                duplicate_ids += int(trade_id in seen_ids)
                seen_ids.add(trade_id)
                if previous_id is not None and trade_id > previous_id + 1:
                    id_gaps += 1
                    missing_id_slots += trade_id - previous_id - 1
                if previous_local is not None:
                    maximum_gap_ns = max(maximum_gap_ns, local - previous_local)
                previous_id = trade_id
                previous_local = local
                quantity = float(row[columns["amount"]])
                price = float(row[columns["price"]])
                nonpositive += int(price <= 0 or quantity <= 0)
                index = (local - start + 99_999_999) // 100_000_000
                if index < 0 or index >= base.height:
                    continue
                side = 0 if row[columns["side"]] == "buy" else 1
                expected[index, side] += quantity
                expected[index, side + 2] += quantity * price
                counts[index] += 1
    actual = base.select("buy_volume", "sell_volume", "buy_notional", "sell_notional").to_numpy()
    flow_valid = base["flow_valid"].to_numpy().astype(bool)
    mismatched = ~np.isclose(actual, expected, rtol=1e-12, atol=1e-8).all(axis=1)
    return {
        "compared_flow_bins": int(flow_valid.sum()),
        "mismatched_flow_bins": int((mismatched & flow_valid).sum()),
        "mismatched_count_bins": int((counts != base["trade_count"].to_numpy()).sum()),
        "trades_in_sample_range": int(counts.sum()),
        "duplicate_trade_ids": duplicate_ids, "nonpositive_trades": nonpositive,
        "id_gap_transitions": id_gaps, "missing_id_slots": missing_id_slots,
        "maximum_intertrade_gap_ns": maximum_gap_ns,
    }


# 对每个采样时刻取最近已收到的独立快照，逐档比较价格与数量。
def validate_snapshot_reference(base: pl.DataFrame, filename: Path) -> dict:
    """输入 C++ 基础表和压缩 25 档快照，返回匹配数量及最早的差异样本。

    快照中间的空窗沿用旧状态。价格和数量以浮点容差比较，时间戳必须精确匹配。
    本过程只读独立快照，盘口仍由正式 C++ 增量回放生成。
    """
    values = base.select(
        *[column for level in range(1, 11) for column in
          [f"bid_price_{level}", f"bid_quantity_{level}", f"ask_price_{level}", f"ask_quantity_{level}"]]
    ).to_numpy()
    timestamps = base["timestamp_ns"].to_numpy()
    valid = base["book_valid"].to_numpy().astype(bool)
    compared = 0
    mismatched = 0
    first_differences = []
    snapshot_rows = 0
    snapshot_times_backwards = 0
    previous_time = -1
    with filename.open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as stream:
            reader = csv.reader(io.TextIOWrapper(stream))
            columns = {name: index for index, name in enumerate(next(reader))}
            fields = [columns[column] for level in range(10) for column in
                      [f"bids[{level}].price", f"bids[{level}].amount", f"asks[{level}].price", f"asks[{level}].amount"]]
            upcoming = next(reader, None)
            latest = None
            for index, timestamp in enumerate(timestamps):
                while upcoming is not None and int(upcoming[columns["local_timestamp"]]) * 1000 <= timestamp:
                    local_time = int(upcoming[columns["local_timestamp"]]) * 1000
                    snapshot_times_backwards += int(local_time < previous_time)
                    previous_time = local_time
                    latest = upcoming
                    snapshot_rows += 1
                    upcoming = next(reader, None)
                if latest is None or not valid[index]:
                    continue
                expected = np.array([float(latest[field]) for field in fields])
                compared += 1
                if not np.allclose(values[index], expected, rtol=0, atol=1e-8):
                    mismatched += 1
                    if len(first_differences) < 5:
                        changed = np.flatnonzero(np.abs(values[index] - expected) > 1e-8)
                        first_differences.append({
                            "timestamp_ns": int(timestamp), "reference_timestamp_ns": previous_time,
                            "field_indices": changed.tolist(),
                            "actual": values[index][changed].tolist(), "reference": expected[changed].tolist(),
                        })
    return {
        "snapshot_rows_consumed": snapshot_rows, "snapshot_times_backwards": snapshot_times_backwards,
        "compared_samples": compared, "mismatched_samples": mismatched,
        "first_differences": first_differences,
    }


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


# 为因果特征和未来标签建立相同的质量区段。
def prepare_quality_state(base: pl.DataFrame) -> pl.DataFrame:
    """输入基础表，返回屏蔽疑点流量并按检测时刻重置区段的表。

    保留引擎区段编号，新增研究区段只在疑点首次可见时生效，保证历史特征不被回写。
    """
    table = base.with_columns(
        pl.col("book_valid").cast(pl.Boolean),
        (pl.col("flow_valid").cast(pl.Boolean) & ~pl.col("trade_sequence_gap")).alias("flow_valid"),
    )
    table = table.with_columns(
        pl.col("segment_id").alias("engine_segment_id"),
        (pl.col("segment_id") + pl.col("trade_sequence_gap").cast(pl.Int64).cum_sum()).alias("segment_id"),
    )
    return table.with_columns(
        pl.when(pl.col("segment_id").diff().fill_null(1) != 0)
        .then(pl.col("timestamp_ns")).otherwise(None).forward_fill().alias("segment_start_ns"),
    )


# 用已有增量回放结果检查快照入口的盘口与同源成交统计。
def validate_base_reference(base: pl.DataFrame, reference: Path) -> dict:
    """输入新基础表及已有 C++ 基础 Parquet，比较双方有效区间的公共字段。

    价格数量使用 1e-8 容差，金额使用 1e-6，不能用此检查声称快照具备逐事件 OFI。
    """
    expected = pl.read_parquet(reference)
    joined = base.join(expected, on="timestamp_ns", how="inner", suffix="_reference", validate="1:1")
    books = joined.filter(pl.col("book_valid") & pl.col("book_valid_reference").cast(pl.Boolean))
    book_fields = [f"{side}_{field}_{level}" for level in range(1, 11)
                   for side in ["bid", "ask"] for field in ["price", "quantity"]]
    book_errors = [(pl.col(column) - pl.col(column + "_reference")).abs() > 1e-8 for column in book_fields]
    flows = joined.filter(pl.col("flow_valid") & pl.col("flow_valid_reference").cast(pl.Boolean))
    flow_errors = []
    for column in ["buy_volume", "sell_volume", "buy_notional", "sell_notional", "trade_count"]:
        tolerance = 1e-6 if "notional" in column else 1e-8
        flow_errors.append((pl.col(column) - pl.col(column + "_reference")).abs() > tolerance)
    report = {
        "compared_books": books.height,
        "mismatched_books": books.filter(pl.any_horizontal(book_errors)).height,
        "compared_flows": flows.height,
        "mismatched_flows": flows.filter(pl.any_horizontal(flow_errors)).height,
    }
    if report["mismatched_books"] or report["mismatched_flows"] or report["compared_books"] == 0:
        raise ValueError(f"快照入口与已有 C++ 基础表不一致 {report}")
    return report


# 将逐条检查结果转换为可读标签，供标准盘口与成交表共同使用。
def build_record_flags(rules: dict[str, pl.Expr]) -> pl.Expr:
    """输入标签名与布尔表达式，返回 List(String) 类型的标签表达式。

    未命中的条件产生空列表，条件为空值时按未命中处理。
    调用方负责为缺失字段设置明确规则，本函数只组织标签，保留所有输入行。
    """
    return pl.concat_list([
        pl.when(condition.fill_null(False)).then(pl.lit(name)).otherwise(None)
        for name, condition in rules.items()
    ]).list.drop_nulls()
