"""按特征组声明公式版本与输出列，每个变量独立发布为全日期Parquet。"""

import polars as pl

from mm_research.features import base, book, returns, trades

KEY_COLUMNS = base.KEY_COLUMNS
QUALITY_POLICY = "null_observations_min_samples_v1"
GROUPS = {
    "base": {"version": base.GROUP_VERSION, "columns": base.BASE_COLUMNS, "model_columns": base.MODEL_COLUMNS},
    "book": {"version": book.GROUP_VERSION, "columns": book.BOOK_FEATURE_COLUMNS, "model_columns": book.BOOK_FEATURE_COLUMNS},
    "trades": {"version": trades.GROUP_VERSION, "columns": trades.TRADE_FEATURE_COLUMNS, "model_columns": trades.TRADE_FEATURE_COLUMNS},
    "returns": {"version": returns.GROUP_VERSION, "columns": returns.RETURN_FEATURE_COLUMNS, "model_columns": returns.RETURN_FEATURE_COLUMNS},
}


# 【非核心】检查配置的组名与公式版本，并把基础组排在最前。
def requested_groups(config: dict) -> list[str]:
    """输入当前研究配置，返回依赖顺序，错误组名或版本明确报错。"""
    for name, version in config["feature_groups"].items():
        if name not in GROUPS or version != GROUPS[name]["version"]:
            raise ValueError(f"特征组名或版本与代码不同 {name} {version}")
    return ["base", *[name for name in config["feature_groups"] if name != "base"]]


# 【核心逻辑18】按所选组构造表达式，实际依赖列从原生表达式提取。
def group_expressions(name: str, config: dict) -> dict[str, pl.Expr]:
    """输入组名和单位、窗口参数，返回该组的字段及公式。

    每列根据自身输入和min_samples产生空值，取消全组五秒预热条件。
    """
    if name == "book":
        return book.book_expressions(config["tick_size"])
    if name == "trades":
        return trades.trade_expressions(config["lot_size"], config["rolling_min_samples"])
    if name == "returns":
        return returns.return_expressions(config["interval_ns"])
    raise ValueError(f"衍生表达式不接受组名 {name}")
