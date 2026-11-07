"""从标准行情独立核对每日特征Parquet中base组的100ms基础计算。"""

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.features.transform import QUALITY_POLICY


# 【非核心】用独立数组计算核对base组采样、成交分桶及因果时间边界。
def validate_feature_intermediates(base: pl.DataFrame, folder: Path, date: str, latency_ns: int,
                           interval_ns: int, lot_size: float, max_book_age_ns: int) -> dict:
    """输入基础表、标准版本目录及日期，返回盘口抽查与全网格成交对照结果。

    参考路径使用searchsorted和bincount，独立于生产的join_asof和group_by。
    """
    calendar = dt.date.fromisoformat(date)
    dates = [str(calendar - dt.timedelta(days=1)), date]
    book_files = [folder / "snapshots" / f"{day}.parquet" for day in dates
                  if (folder / "snapshots" / f"{day}.parquet").exists()]
    trade_files = [folder / "trades" / f"{day}.parquet" for day in dates
                   if (folder / "trades" / f"{day}.parquet").exists()]
    grid = base["timestamp_ns"].to_numpy()
    book_samples = validate_standard_books(base, book_files, grid, latency_ns, max_book_age_ns)
    visible_trades = validate_standard_trades(base, trade_files, grid, latency_ns, interval_ns, lot_size)
    if base.filter(pl.col("last_available_timestamp_ns") > pl.col("timestamp_ns")).height:
        raise ValueError("采样含未来行情")
    return {
        "rows": base.height, "book_samples_checked": book_samples,
        "trade_bins_checked": base.height, "visible_trades": visible_trades,
        "quality_policy": QUALITY_POLICY, "market_data_latency_ns": latency_ns,
    }


# 【非核心】抽查最近可见快照，比较十档价量和盘口有效性。
def validate_standard_books(base: pl.DataFrame, book_files: list[Path],
                            grid: np.ndarray, latency_ns: int, max_book_age_ns: int) -> int:
    """返回检查的盘口样本数，同时核对全部十档值与盘口有效性。"""
    fields = [f"{side}_{field}_{level}" for level in range(1, 11)
              for side in ["bid", "ask"] for field in ["price", "quantity"]]
    books = pl.read_parquet(book_files, columns=["exchange_timestamp_ns", *fields])
    order = np.argsort(books["exchange_timestamp_ns"].to_numpy(), kind="stable")
    times = books["exchange_timestamp_ns"].to_numpy()[order] + latency_ns
    book_values = books.select(fields).to_numpy()[order]
    samples = np.arange(0, base.height, 997)
    latest = np.searchsorted(times, grid[samples], side="right") - 1
    available = latest >= 0
    sample_indices = samples[available]
    expected = book_values[latest[available]]
    actual = base.select(fields).to_numpy()[sample_indices]
    expected_valid = np.isfinite(expected).all(axis=1) & (expected > 0).all(axis=1)
    expected_valid &= expected[:, 0] < expected[:, 2]
    for level in range(9):
        expected_valid &= expected[:, level * 4] > expected[:, (level + 1) * 4]
        expected_valid &= expected[:, level * 4 + 2] < expected[:, (level + 1) * 4 + 2]
    expected_valid &= grid[sample_indices] - times[latest[available]] <= max_book_age_ns
    expected[~expected_valid] = np.nan
    if not np.allclose(actual, expected, atol=1e-10, rtol=0, equal_nan=True):
        raise ValueError("独立盘口采样值不一致")
    if not np.array_equal(base["book_valid"].to_numpy()[sample_indices], expected_valid):
        raise ValueError("独立盘口有效性不一致")
    if base["book_valid"].to_numpy()[samples[~available]].any():
        raise ValueError("尚无可见快照时盘口应无效")
    return len(samples)


# 【非核心】独立分桶原始成交，逐网格核对笔数、数量和金额。
def validate_standard_trades(base: pl.DataFrame, trade_files: list[Path],
                             grid: np.ndarray, latency_ns: int, interval_ns: int,
                             lot_size: float) -> int:
    """返回采样范围内总成交笔数，整数Lot避免浮点数量累积误差。"""
    trades = pl.read_parquet(trade_files, columns=["exchange_timestamp_ns", "price", "quantity", "side", "quality_flags"])
    available_times = trades["exchange_timestamp_ns"].to_numpy() + latency_ns
    indices = (available_times - grid[0] + interval_ns - 1) // interval_ns
    within = (indices >= 0) & (indices < base.height)
    counts = np.bincount(indices[within], minlength=base.height)
    valid = trades.select(
        (pl.col("price").is_finite() & (pl.col("price") > 0)
         & pl.col("quantity").is_finite() & (pl.col("quantity") > 0)
         & pl.col("side").is_in(["buy", "sell"])
         & ~pl.col("quality_flags").list.eval(
             pl.element().is_in(["content_conflict", "off_tick", "off_lot", "missing_trade_id"])
         ).list.any()).fill_null(False)
    ).to_series().to_numpy()
    invalid_counts = np.bincount(indices[within & ~valid], minlength=base.height)
    flow_valid = invalid_counts == 0
    if len(trade_files) == 1:
        flow_valid &= grid >= grid[0] + latency_ns + interval_ns
    if not np.array_equal(flow_valid, base["flow_valid"].to_numpy()):
        raise ValueError("独立成交窗口质量不一致")
    expected_counts = counts.astype(float)
    expected_counts[~flow_valid] = np.nan
    if not np.allclose(expected_counts, base["trade_count"].to_numpy(), equal_nan=True):
        raise ValueError("独立逐网格成交计数不一致")
    quantity = trades["quantity"].to_numpy()
    price = trades["price"].to_numpy()
    for side in ["buy", "sell"]:
        chosen = within & valid & (trades["side"].to_numpy() == side)
        lots = np.rint(quantity[chosen] / lot_size)
        volume = np.bincount(indices[chosen], weights=lots, minlength=base.height) * lot_size
        notional = np.bincount(indices[chosen], weights=price[chosen] * quantity[chosen], minlength=base.height).astype(float)
        volume[~flow_valid] = np.nan
        notional[~flow_valid] = np.nan
        if not np.allclose(volume, base[f"{side}_volume"].to_numpy(), atol=1e-10, rtol=0, equal_nan=True):
            raise ValueError("独立成交数量不一致")
        if not np.allclose(notional, base[f"{side}_notional"].to_numpy(), atol=1e-6, rtol=1e-12, equal_nan=True):
            raise ValueError("独立成交金额不一致")
    return int(counts.sum())
