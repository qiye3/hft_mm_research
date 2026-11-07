"""Gen001买卖压力短长窗口差，公式来自用户的因子构造文档。

盘口压力、主动成交压力及两者背离分别计算10点与50点均值差。
窗口始终按时间位置计数，缺失值由min_samples处理，跨午夜连续计算。
"""

import polars as pl

INPUT_COLUMNS = ["imbalance_1", "imbalance_5", "imbalance_10", "trade_imbalance_100ms", "trade_imbalance_1s"]
VERSION = "v1"


# 【核心逻辑28】为六个压力序列构造原生短长窗口均值差表达式。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """输入短长窗口及最少有效样本数，返回六个具名因子的表达式。

    两个均值分别检查有效样本，任一不足则差值为空，经济含义正值表示近期买方压力增强。
    """
    short = config["short_window"]
    long = config["long_window"]
    if not (1 <= config["short_min_samples"] <= short < long
            and 1 <= config["long_min_samples"] <= long):
        raise ValueError("Gen001窗口和min_samples不合法")
    sources = {
        "book_obi_l1": pl.col("imbalance_1"),
        "book_obi_l5": pl.col("imbalance_5"),
        "book_obi_l10": pl.col("imbalance_10"),
        "trade_aggr_imb_100ms": pl.col("trade_imbalance_100ms"),
        "trade_aggr_imb_1s": pl.col("trade_imbalance_1s"),
        "cross_trade_book_gap": pl.col("trade_imbalance_1s") - pl.col("imbalance_5"),
    }
    expressions = {}
    for name, source in sources.items():
        source = source.fill_nan(None)
        short_mean = source.rolling_mean(short, min_samples=config["short_min_samples"])
        long_mean = source.rolling_mean(long, min_samples=config["long_min_samples"])
        expressions[f"{name}_mean_diff_{short}_{long}"] = short_mean - long_mean
    return expressions
