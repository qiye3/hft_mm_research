"""Gen003 盘口深度结构与形态，公式与用户构造文档逐项对应。"""

import polars as pl

from mm_research.factors.operators import rolling_factors
from mm_research.factors.sources import shape_sources

VERSION = "seconds_v1"


# 【核心逻辑47】同批展开各里层序列与文档指定窗口，供分块并行计算。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """输入缺失有效比例和价格步长，外层窗口单位为秒，返回完整因子名及其Polars表达式。"""
    return rolling_factors(
        shape_sources(config["tick_size"]), config["aggregation_windows_s"], "mean", config["min_fraction"],
    )
