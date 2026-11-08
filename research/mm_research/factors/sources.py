"""因子文档中的里层变换。所有公式只依赖当前和历史可见Features。"""

import polars as pl

from mm_research.features.expressions import normalized_imbalance


# 【核心逻辑42】复用已生成的盘口、主动成交失衡，并构造两者差值。
def pressure_sources() -> dict[str, pl.Expr]:
    """返回六个无量纲压力序列，已知无成交的失衡口径沿用Features。"""
    return {
        "book_obi_l1": pl.col("imbalance_1"),
        "book_obi_l5": pl.col("imbalance_5"),
        "book_obi_l10": pl.col("imbalance_10"),
        "trade_aggr_imb_100ms": pl.col("trade_imbalance_100ms"),
        "trade_aggr_imb_1s": pl.col("trade_imbalance_1s"),
        "cross_trade_book_gap": pl.col("trade_imbalance_1s") - pl.col("imbalance_5"),
    }


# 【核心逻辑49】累计指定深度的数量，任一所需档位缺失时保持null。
def depth_quantity(side: str, depth: int) -> pl.Expr:
    """返回BTC数量表达式，禁止部分档位缺失时按零补齐。"""
    return pl.sum_horizontal(
        [pl.col(f"{side}_quantity_{level}") for level in range(1, depth + 1)], ignore_nulls=False,
    )


# 【核心逻辑48】构造盘口集中度、加权距离及双侧形态差异。
def shape_sources(tick_size: float) -> dict[str, pl.Expr]:
    """数量权重距离单位Tick，集中度无量纲，输入为前十档快照。"""
    expressions = {}
    for side in ["bid", "ask"]:
        total = depth_quantity(side, 10)
        for depth in [1, 5]:
            expressions[f"book_{side}_share_l{depth}_l10"] = depth_quantity(side, depth) / total
        distances = []
        for level in range(1, 11):
            difference = pl.col(f"{side}_price_{level}") - pl.col(f"{side}_price_1")
            if side == "bid":
                difference = -difference
            distances.append(difference / tick_size * pl.col(f"{side}_quantity_{level}"))
        expressions[f"book_{side}_weighted_dist_l10"] = pl.sum_horizontal(distances, ignore_nulls=False) / total
    for depth in [1, 5]:
        expressions[f"book_near_conc_gap_l{depth}_l10"] = (
            expressions[f"book_bid_share_l{depth}_l10"] - expressions[f"book_ask_share_l{depth}_l10"]
        )
    expressions["book_dist_asym_l10"] = (
        expressions["book_bid_weighted_dist_l10"] - expressions["book_ask_weighted_dist_l10"]
    )
    return expressions


# 【核心逻辑51】按文档构造主动成交数量、金额和笔数，保留已知零成交。
def activity_sources() -> dict[str, pl.Expr]:
    """返回BTC、USDT和笔数三类序列，异常成交窗口沿用上游null。"""
    expressions = {"trade_count_100ms": pl.col("trade_count")}
    for unit, buy, sell in [("qty", "buy_volume", "sell_volume"), ("notional", "buy_notional", "sell_notional")]:
        expressions[f"trade_buy_{unit}_100ms"] = pl.col(buy)
        expressions[f"trade_sell_{unit}_100ms"] = pl.col(sell)
        expressions[f"trade_net_{unit}_100ms"] = pl.col(buy) - pl.col(sell)
        expressions[f"trade_total_{unit}_100ms"] = pl.col(buy) + pl.col(sell)
    return expressions


# 【核心逻辑53】从100ms收益生成路径、半方差、移动和转向计数。
def price_sources() -> dict[str, pl.Expr]:
    """收益单位bps，转向要求相邻两条收益均有效且非零。"""
    returns = pl.col("return_100ms_bps")
    positive = returns.clip(lower_bound=0)
    negative = (-returns).clip(lower_bound=0)
    return {
        "px_ret_100ms": returns,
        "px_abs_ret_100ms": returns.abs(),
        "px_pos_ret_100ms": positive,
        "px_neg_ret_100ms": negative,
        "px_sq_ret_100ms": returns.pow(2),
        "px_pos_sq_ret_100ms": positive.pow(2),
        "px_neg_sq_ret_100ms": negative.pow(2),
        "px_move_flag_100ms": (returns != 0).cast(pl.Float64),
        "px_turn_flag_100ms": pl.when(returns.is_not_null() & returns.shift(1).is_not_null()).then(
            ((returns != 0) & (returns.shift(1) != 0) & (returns.sign() != returns.shift(1).sign())).cast(pl.Float64)
        ).otherwise(None),
    }


# 【核心逻辑55】对相邻100ms排行深度做差，刻画可见流动性的增减。
def liquidity_sources() -> dict[str, pl.Expr]:
    """输出BTC数量变化，比较排行档位总量，保留跨缺失位置的null。"""
    expressions = {}
    for depth in [1, 5, 10]:
        buy = depth_quantity("bid", depth).diff()
        sell = depth_quantity("ask", depth).diff()
        expressions[f"book_bid_depth_l{depth}_delta_100ms"] = buy
        expressions[f"book_ask_depth_l{depth}_delta_100ms"] = sell
        expressions[f"book_net_depth_l{depth}_delta_100ms"] = buy - sell
    return expressions


# 【核心逻辑57】以主动成交数量或失衡定义压力，构造同期吸收和反向变化。
def impact_sources() -> dict[str, pl.Expr]:
    """本轮只构造成交类压力，事件OFI类由用户明确排除。"""
    pressure = {
        "trade_qty": (pl.col("buy_volume") - pl.col("sell_volume"))
        / (depth_quantity("bid", 5) + depth_quantity("ask", 5)),
        "trade_imb": pl.col("trade_imbalance_100ms"),
    }
    returns = pl.col("return_100ms_bps")
    expressions = {}
    for name, values in pressure.items():
        aligned = values.sign() * returns
        expressions[f"impact_{name}_aligned_100ms"] = aligned
        expressions[f"impact_{name}_absorb_100ms"] = values.abs() * (returns == 0).cast(pl.Float64)
        expressions[f"impact_{name}_opposed_100ms"] = (-aligned).clip(lower_bound=0)
    return expressions


# 【核心逻辑59】构造最新成交压力和压力加速度相对三个盘口深度的背离。
def divergence_sources() -> dict[str, pl.Expr]:
    """返回六种无量纲背离，排除本轮未授权计算的OFI方向分支。"""
    pressure = {
        "trade_aggr_imb_100ms": pl.col("trade_imbalance_100ms"),
        "trade_aggr_imb_accel_100ms": pl.col("trade_imbalance_100ms") - pl.col("trade_imbalance_1s"),
    }
    return {
        f"cross_{name}_book_obi_l{depth}_gap": values - pl.col(f"imbalance_{depth}")
        for name, values in pressure.items()
        for depth in [1, 5, 10]
    }


# 【核心逻辑61】读取按冻结训练阈值生成的分层成交桶，构造各层数量和失衡。
def size_sources() -> dict[str, pl.Expr]:
    """返回小中大三层的主动买卖量与失衡，已知空桶失衡为零。"""
    expressions = {}
    for size in ["small", "mid", "large"]:
        buy = pl.col(f"trade_{size}_buy_qty_100ms")
        sell = pl.col(f"trade_{size}_sell_qty_100ms")
        expressions[f"trade_{size}_buy_qty_100ms"] = buy
        expressions[f"trade_{size}_sell_qty_100ms"] = sell
        expressions[f"trade_{size}_imb_100ms"] = normalized_imbalance(buy, sell, empty_value=0)
    return expressions
