"""Gen012使用前十档显示深度模拟固定名义额执行成本，深度不足为空。"""
import polars as pl
from mm_research.factors.operators import rolling_factors
from mm_research.factors.sources import depth_quantity

VERSION = "seconds_v1"


# 【核心逻辑66】按价格顺序分配目标数量，末档只消耗剩余量，输出理论VWAP。
def execution_vwap(side: str, quantity: pl.Expr) -> pl.Expr:
    """输入盘口方向和BTC目标量，十档完整且足量时返回USDT均价。"""
    allocated = pl.lit(0.0)
    amounts = []
    for level in range(1, 11):
        size = pl.col(f"{side}_quantity_{level}")
        taken = pl.min_horizontal((quantity - allocated).clip(lower_bound=0), size)
        amounts.append(taken * pl.col(f"{side}_price_{level}"))
        allocated = allocated + size
    return pl.when((quantity > 0) & (depth_quantity(side, 10) >= quantity)).then(
        pl.sum_horizontal(amounts, ignore_nulls=False) / quantity,
    ).otherwise(None)


# 【核心逻辑65】从当前Mid换算三档名义额，分别计算买卖成本及方向差。
def source_expressions(config: dict) -> dict[str, pl.Expr]:
    """输出5k、25k、100k USDT执行成本的秒级滚动均值，单位bps。"""
    mid = pl.col("mid_price")
    sources = {}
    for name, notional in [("5k", 5000), ("25k", 25000), ("100k", 100000)]:
        buy = (execution_vwap("ask", notional / mid) / mid - 1) * 10000
        sell = (1 - execution_vwap("bid", notional / mid) / mid) * 10000
        sources[f"book_exec_buy_cost_{name}"] = buy
        sources[f"book_exec_sell_cost_{name}"] = sell
        sources[f"book_exec_gap_cost_{name}"] = buy - sell
    return sources


# 【核心逻辑64】共享已计算的里层状态，再计算各秒级窗口以控制中间内存。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """每个里层序列只生成一次，返回四档窗口的Float64表达式。"""
    sources = {name: pl.col(name) for name in source_expressions(config)}
    return rolling_factors(sources, (5, 10, 30, 100), 'mean', config['min_fraction'])
