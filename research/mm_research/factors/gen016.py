"""Gen016当前成交VWAP相对桶末报价与上一桶Mid的偏离，单位bps。"""
import polars as pl
from mm_research.factors.operators import book_transition_allowed, rolling_factors

VERSION = "seconds_v2"


# 【核心逻辑72】仅在有成交时计算均价，空桶均价保持为空。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """九类价差由成交金额与数量重建，输出四档秒窗口均值。"""
    buy_volume, sell_volume = pl.col("buy_volume"), pl.col("sell_volume")
    buy = pl.col("buy_notional") / pl.when(buy_volume > 0).then(buy_volume)
    sell = pl.col("sell_notional") / pl.when(sell_volume > 0).then(sell_volume)
    total = (pl.col("buy_notional") + pl.col("sell_notional")) / pl.when(buy_volume + sell_volume > 0).then(buy_volume + sell_volume)
    mid = pl.col("mid_price")
    previous = pl.when(book_transition_allowed()).then(mid.shift(1))
    sources = {
        "trade_buy_vwap_mid_dev_bps_100ms": (buy - mid) / mid * 10000,
        "trade_sell_vwap_mid_dev_bps_100ms": (sell - mid) / mid * 10000,
        "trade_all_vwap_mid_dev_bps_100ms": (total - mid) / mid * 10000,
        "trade_buy_vwap_ask_dev_bps_100ms": (buy - pl.col("ask_price_1")) / mid * 10000,
        "trade_sell_vwap_bid_dev_bps_100ms": (sell - pl.col("bid_price_1")) / mid * 10000,
        "trade_buy_sell_vwap_gap_bps_100ms": (buy - sell) / mid * 10000,
        "trade_buy_vwap_prev_mid_dev_bps_100ms": (buy - previous) / previous * 10000,
        "trade_sell_vwap_prev_mid_dev_bps_100ms": (sell - previous) / previous * 10000,
        "trade_all_vwap_prev_mid_dev_bps_100ms": (total - previous) / previous * 10000,
    }
    return rolling_factors(sources, config["aggregation_windows_s"], "mean", config["min_fraction"])
