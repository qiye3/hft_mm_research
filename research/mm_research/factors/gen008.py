"""Gen008 成交冲击与价格吸收，排除OFI，公式与用户构造文档逐项对应。"""

import polars as pl

from mm_research.factors.operators import rolling_factors
from mm_research.factors.sources import impact_sources

VERSION = "seconds_v1"


# 【核心逻辑56】同批展开各里层序列与文档指定窗口，供分块并行计算。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """输入缺失有效比例和价格步长，外层窗口单位为秒，返回完整因子名及其Polars表达式。"""
    return rolling_factors(
        impact_sources(), (5, 10, 20, 50), "mean", config["min_fraction"],
    )
