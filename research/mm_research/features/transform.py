"""按依赖计算Features组，base与衍生列共同进入每日特征Parquet。

各组声明版本、输出列和有效性。新增组只读取所需base列，依赖base定义的标签独立保存。
"""

import polars as pl

from mm_research.features import base, book, returns, trades
from mm_research.features.expressions import rolling_sum_in_segment

QUALITY_POLICY = "grouped_book_trade_quality_v1"
KEY_COLUMNS = base.KEY_COLUMNS
GROUPS = {
    "base": {
        "version": base.GROUP_VERSION,
        "columns": base.BASE_COLUMNS,
        "model_columns": base.MODEL_COLUMNS,
        "validity": "base_feature_valid",
        "dependencies": [],
    },
    "book": {
        "version": book.GROUP_VERSION,
        "columns": book.BOOK_FEATURE_COLUMNS,
        "model_columns": book.BOOK_FEATURE_COLUMNS,
        "validity": "book_feature_valid",
        "dependencies": ["base"],
    },
    "trades": {
        "version": trades.GROUP_VERSION,
        "columns": trades.TRADE_FEATURE_COLUMNS,
        "model_columns": trades.TRADE_FEATURE_COLUMNS,
        "validity": "trade_feature_valid",
        "dependencies": ["base"],
    },
    "returns": {
        "version": returns.GROUP_VERSION,
        "columns": returns.RETURN_FEATURE_COLUMNS,
        "model_columns": returns.RETURN_FEATURE_COLUMNS,
        "validity": "return_feature_valid",
        "dependencies": ["base"],
    },
}


# 【非核心】检查组名与公式版本，确定包含base依赖的执行顺序。
def requested_groups(config: dict) -> list[str]:
    """输入Features配置，返回确定性的依赖顺序，版本不匹配时明确报错。"""
    selected = config["feature_groups"]
    unknown = set(selected) - GROUPS.keys()
    if unknown:
        raise ValueError(f"未知特征组 {sorted(unknown)}")
    for name, version in selected.items():
        if version != GROUPS[name]["version"]:
            raise ValueError(f"{name}组配置版本{version}与代码版本{GROUPS[name]['version']}不同")
    return ["base", *[name for name in selected if name != "base"]]


# 【非核心】整理单组输出列和依赖版本，供增量更新与元数据核对。
def group_definition(name: str) -> dict:
    """返回该组版本、输出、模型可选列和直接依赖的版本。"""
    definition = dict(GROUPS[name])
    definition["dependencies"] = {dependency: GROUPS[dependency]["version"]
                                  for dependency in GROUPS[name]["dependencies"]}
    return definition


# 【核心逻辑16】调用所选组的特征公式，并按该组依赖判断样本有效性。
def build_feature_group(name: str, table: pl.DataFrame, config: dict) -> pl.DataFrame:
    """输入base列与单组名称，返回样本键、该组输出和独立有效性标记。

    book无需滚动预热，trades仅检查成交历史，returns仅检查盘口历史。
    """
    if name == "book":
        values = book.add_book_features(table, config["tick_size"])
        valid = pl.col("book_valid").cast(pl.Boolean)
    elif name == "trades":
        values = table.with_columns(pl.col("trade_segment_id").alias("segment_id"))
        values = trades.add_trade_features(values, config["lot_size"])
        valid = rolling_sum_in_segment(pl.col("flow_valid").cast(pl.Int32), 50) == 50
    elif name == "returns":
        values = table.with_columns(pl.col("book_segment_id").alias("segment_id"))
        values = returns.add_return_features(values)
        valid = pl.col("book_valid").cast(pl.Boolean)
        valid = valid & (rolling_sum_in_segment(pl.col("book_valid").cast(pl.Int32), 51) == 51)
    else:
        raise ValueError(f"衍生组计算不接受{name}")
    definition = GROUPS[name]
    finite = pl.all_horizontal([
        pl.col(column).is_not_null() & pl.col(column).is_finite() for column in definition["columns"]
    ])
    return values.select(
        *KEY_COLUMNS, *definition["columns"],
        (valid & finite).fill_null(False).alias(definition["validity"]),
    )
