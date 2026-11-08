"""Gen002 买卖压力短长窗口差，公式与用户构造文档逐项对应。"""

import polars as pl

from mm_research.factors.operators import rolling_factors
from mm_research.factors.sources import pressure_sources

VERSION = "seconds_v1"


# 【核心逻辑46】同批展开各里层序列与文档指定窗口，供分块并行计算。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """输入缺失有效比例和价格步长，外层窗口单位为秒，返回完整因子名及其Polars表达式。"""
    return rolling_factors(
        pressure_sources(), ((3, 10), (5, 20), (5, 50), (10, 50), (10, 100), (20, 100)), "mean_diff", config["min_fraction"],
    )
