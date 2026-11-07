"""Polars原生分布、逐日IC和分组收益，输入数值缺失统一为null。"""

import polars as pl


# 【核心逻辑31】分别统计因子自身有效性、零值率与数值分布。
def distribution_statistics(table: pl.DataFrame) -> dict:
    """输入含value的表，返回分布字典，零值率以有效因子样本为分母。"""
    values = table.lazy().select(pl.col("value").fill_nan(None))
    return values.select(
        pl.len().alias("rows"), pl.col("value").count().alias("valid_rows"),
        pl.col("value").is_null().mean().alias("missing_rate"),
        (pl.col("value") == 0).sum().truediv(pl.col("value").count()).alias("zero_rate"),
        pl.col("value").mean().alias("mean"), pl.col("value").std().alias("std"),
        pl.col("value").min().alias("min"), pl.col("value").max().alias("max"),
        *[pl.col("value").quantile(quantile, interpolation="linear").alias(name)
          for quantile, name in [(0.01, "q01"), (0.05, "q05"), (0.5, "median"), (0.95, "q95"), (0.99, "q99")]],
    ).with_columns(pl.all().exclude("rows", "valid_rows").fill_nan(None)).collect().row(0, named=True)


# 【核心逻辑32】按因子和标签共同有效样本计算Pearson及平均排名Rank IC。
def correlation_statistics(table: pl.DataFrame) -> dict:
    """输入关联表，返回样本损耗及两个IC，常数或不足两条样本时给出原因。"""
    valid_factor = pl.col("value").is_not_null() & pl.col("value").is_finite()
    valid_target = pl.col("label_valid").fill_null(False) & pl.col("target").is_not_null() & pl.col("target").is_finite()
    counts = table.select(
        pl.len().alias("rows"), valid_factor.sum().alias("factor_valid_rows"),
        valid_target.sum().alias("label_valid_rows"), (valid_factor & valid_target).sum().alias("paired_rows"),
    ).row(0, named=True)
    paired = table.filter(valid_factor & valid_target)
    metrics = paired.select(
        pl.col("value").n_unique().alias("factor_unique"),
        pl.col("target").n_unique().alias("target_unique"),
        pl.corr("value", "target").alias("ic"),
        pl.corr("value", "target", method="spearman").alias("rank_ic"),
    ).with_columns(pl.col("ic", "rank_ic").fill_nan(None)).row(0, named=True)
    reason = "ok"
    if counts["paired_rows"] < 2:
        reason = "insufficient_pairs"
    elif metrics["factor_unique"] < 2:
        reason = "constant_factor"
    elif metrics["target_unique"] < 2:
        reason = "constant_target"
    return {**counts, "ic": metrics["ic"], "rank_ic": metrics["rank_ic"], "reason": reason}


# 【核心逻辑33】按分位边界对相同因子值一致归组，计算组内未来收益。
def grouped_returns(table: pl.DataFrame, groups: int) -> pl.DataFrame:
    """输入关联表与目标组数，返回实际组数、样本量、因子均值与收益均值。

    分位边界使用线性插值，重复边界合并。边界相同的值进入同一组，不用时间顺序拆分重复值。
    """
    paired = table.filter(
        pl.col("label_valid").fill_null(False) & pl.col("value").is_finite() & pl.col("target").is_finite(),
    )
    if paired.is_empty():
        return pl.DataFrame(schema={"group": pl.Int32, "samples": pl.UInt32, "factor_mean": pl.Float64,
                                    "mean_return_bps": pl.Float64, "actual_groups": pl.Int32})
    cutoffs = paired.select([
        pl.col("value").quantile(index / groups, interpolation="linear").alias(f"q{index}")
        for index in range(1, groups)
    ]).row(0)
    boundaries = sorted(set(cutoffs))
    assignment = pl.lit(1)
    for cutoff in boundaries:
        assignment = assignment + (pl.col("value") > cutoff).cast(pl.Int32)
    result = paired.with_columns(assignment.cast(pl.Int32).alias("group")).group_by("group").agg(
        pl.len().alias("samples"), pl.col("value").mean().alias("factor_mean"),
        pl.col("target").mean().alias("mean_return_bps"),
    ).sort("group")
    return result.with_columns(pl.lit(result.height, dtype=pl.Int32).alias("actual_groups"))
