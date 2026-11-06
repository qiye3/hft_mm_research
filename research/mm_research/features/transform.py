"""组织分类特征计算，统一判断研究样本有效性。

build_feature_table 输入连续 100ms 基础表，输出源字段、分类特征与 feature_valid。
FEATURE_COLUMNS 保留事件版十九列顺序，快照版选择十六列，标签独立生成。
"""

import polars as pl

from mm_research.data.quality import prepare_quality_state
from mm_research.features.book import BOOK_FEATURE_COLUMNS, add_book_features
from mm_research.features.expressions import rolling_sum_in_segment
from mm_research.features.order_flow import ORDER_FLOW_FEATURE_COLUMNS, add_order_flow_features
from mm_research.features.returns import RETURN_FEATURE_COLUMNS, add_return_features
from mm_research.features.trades import TRADE_FEATURE_COLUMNS, add_trade_features

FEATURE_COLUMNS = BOOK_FEATURE_COLUMNS + ORDER_FLOW_FEATURE_COLUMNS + TRADE_FEATURE_COLUMNS + RETURN_FEATURE_COLUMNS
SNAPSHOT_FEATURE_COLUMNS = BOOK_FEATURE_COLUMNS + TRADE_FEATURE_COLUMNS + RETURN_FEATURE_COLUMNS


# 统一准备质量区段，依次构造各类特征并判断五秒预热。
def build_feature_table(base: pl.DataFrame, tick_size: float, interval_ns: int,
                        lot_size: float = 0.001, feature_columns: list[str] | None = None) -> pl.DataFrame:
    """输入基础表、价格数量步长及所选特征，返回保留源字段的完整特征表。

    网格固定 100ms，各窗口左开右闭且不跨质量区段。五秒完整流量和历史价格共同
    决定预热，特征可用性独立于未来标签，已有生产调用继续使用本入口。
    """
    if interval_ns != 100_000_000:
        raise ValueError("当前特征版本固定使用 100ms 网格")
    selected_columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    table = prepare_quality_state(base)
    table = add_book_features(table, tick_size)
    table = add_order_flow_features(table)
    table = add_trade_features(table, lot_size)
    table = add_return_features(table)
    valid_bins = rolling_sum_in_segment(pl.col("flow_valid").cast(pl.Int32), 50)
    finite_features = pl.all_horizontal([
        pl.col(column).is_not_null() & pl.col(column).is_finite() for column in selected_columns
    ])
    return table.with_columns(
        (pl.col("book_valid") & (valid_bins == 50)
         & (pl.col("timestamp_ns") - pl.col("segment_start_ns") >= 5_000_000_000)
         & finite_features).fill_null(False).alias("feature_valid"),
    )
