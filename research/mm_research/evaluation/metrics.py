"""Polars原生分布、逐日IC和分组收益，输入数值缺失统一为null。"""

import polars as pl


# 【核心逻辑114】分别统计因子自身有效性、零值率与数值分布。
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


# 【核心逻辑117】按因子和标签共同有效样本计算Pearson及平均排名Rank IC。
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


# 【核心逻辑118】使用冻结训练阈值分组，重复边界时显式采用负零正三组。
def grouped_returns(table: pl.DataFrame, groups: int, boundaries: dict) -> pl.DataFrame:
    """输入关联表与训练阈值，返回样本量和条件未来收益，禁止在测试日重拟阈值。"""
    paired = table.filter(
        pl.col("label_valid").fill_null(False) & pl.col("value").is_finite() & pl.col("target").is_finite(),
    )
    if boundaries["mode"] == "sign":
        assignment = pl.when(pl.col("value") < 0).then(1).when(pl.col("value") == 0).then(2).otherwise(3)
        requested_groups = 3
    elif boundaries["mode"] == "quantile":
        assignment = pl.lit(1)
        for cutoff in boundaries["cutoffs"]:
            assignment = assignment + (pl.col("value") > cutoff).cast(pl.Int32)
        requested_groups = groups
    else:
        raise ValueError("未知冻结分组模式")
    result = paired.with_columns(assignment.cast(pl.Int32).alias("group")).group_by("group").agg(
        pl.len().alias("samples"), pl.col("value").mean().alias("factor_mean"),
        pl.col("target").mean().alias("mean_return_bps"),
    ).sort("group")
    return result.with_columns(
        pl.lit(result.height, dtype=pl.Int32).alias("actual_groups"),
        pl.lit(requested_groups, dtype=pl.Int32).alias("requested_groups"),
    )
