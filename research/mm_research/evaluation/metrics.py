"""Polars原生分布、逐日IC和分组收益，输入数值缺失统一为null。"""

import math

import numpy as np
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
    # 配对样本先物化一次，常数判定只扫描极值，避免每次相关计算额外构造唯一值集合。
    metrics = paired.select(
        (pl.col('value').min() == pl.col('value').max()).alias('constant_factor'),
        (pl.col('target').min() == pl.col('target').max()).alias('constant_target'),
        pl.corr('value', 'target').fill_nan(None).alias('ic'),
        pl.corr('value', 'target', method='spearman').fill_nan(None).alias('rank_ic'),
    ).row(0, named=True)
    reason = "ok"
    if counts["paired_rows"] < 2:
        reason = "insufficient_pairs"
    elif metrics['constant_factor']:
        reason = "constant_factor"
    elif metrics['constant_target']:
        reason = "constant_target"
    return {**counts, 'ic': metrics['ic'], 'rank_ic': metrics['rank_ic'], 'reason': reason}


# 【核心逻辑118】使用冻结训练阈值分组，重复边界时显式采用负零正三组。
def grouped_returns(table: pl.DataFrame, groups: int, boundaries: dict) -> pl.DataFrame:
    """输入关联表与训练阈值，返回样本量和条件未来收益，禁止在测试日重拟阈值。"""
    paired = table.lazy().filter(
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
        pl.len().cast(pl.Int32).alias("actual_groups"),
        pl.lit(requested_groups, dtype=pl.Int32).alias("requested_groups"),
    ).collect()


# 【核心逻辑120】训练期日均IC唯一确定因子方向，验证期与测试期沿用，不重新估计。
def train_ic_direction(daily_ic: pl.DataFrame) -> pl.DataFrame:
    """输入逐日IC表，按周期和采样返回训练期方向，正1负-1，零或不可计算为null。"""
    return (daily_ic.filter(pl.col("split") == "train").group_by("horizon_ns", "sampling")
            .agg(pl.col("ic").drop_nulls().mean().alias("train_ic_mean"))
            .select(
                "horizon_ns", "sampling",
                pl.when(pl.col("train_ic_mean") > 0).then(1)
                .when(pl.col("train_ic_mean") < 0).then(-1)
                .otherwise(None).cast(pl.Int32).alias("train_ic_direction"),
            ))


# 【核心逻辑121】方向一致率只统计有限日IC且零IC不计一致，符号组价差要求当日两端点组同时存在。
def oriented_statistics(daily_ic: pl.DataFrame, returns: pl.DataFrame,
                        directions: pl.DataFrame) -> pl.DataFrame:
    """输入逐日IC、分组收益和训练方向，返回各split×周期×采样的方向一致率与sign组价差。"""
    consistent = ((pl.col("train_ic_direction") == 1) & (pl.col("ic") > 0)) | \
                 ((pl.col("train_ic_direction") == -1) & (pl.col("ic") < 0))
    consistency = (daily_ic.join(directions, on=["horizon_ns", "sampling"], how="left")
                   .group_by("split", "horizon_ns", "sampling")
                   .agg(pl.col("ic").is_not_null().sum().alias("direction_days"),
                        consistent.fill_null(False).sum().alias("consistent_days"),
                        pl.col("train_ic_direction").first())
                   .select(
                       "split", "horizon_ns", "sampling", "train_ic_direction",
                       pl.when(pl.col("train_ic_direction").is_not_null() & (pl.col("direction_days") > 0))
                       .then(pl.col("consistent_days") / pl.col("direction_days"))
                       .otherwise(None).alias("direction_consistency_ratio"),
                   ))
    spreads = (endpoint_spread_days(returns, 3).group_by("split", "horizon_ns", "sampling")
               .agg(pl.col("daily_spread").drop_nulls().mean().alias("sign_group_spread_bps"),
                    pl.col("daily_spread").is_not_null().sum().alias("sign_spread_days")))
    return consistency.join(spreads, on=["split", "horizon_ns", "sampling"], how="left")


# 【核心逻辑122】按UTC交易日有放回重采样，固定种子2000次估计日均量的95%区间。
def bootstrap_interval(values: list[float | None]) -> dict:
    """输入按日期排列的日指标，返回95%上下界和参与重采样的有效日数，不足两日返回null。"""
    observations = np.asarray([value for value in values if value is not None], dtype=float)
    if len(observations) < 2:
        return {"ci95_low": None, "ci95_high": None, "valid_days": len(observations)}
    generator = np.random.default_rng(20261108)
    means = np.empty(2000, dtype=np.float64)
    for begin in range(0, 2000, 100):
        stop = min(begin + 100, 2000)
        draws = generator.integers(0, len(observations), size=(stop - begin, len(observations)))
        means[begin:stop] = observations[draws].mean(axis=1)
    lower, upper = np.quantile(means, [.025, .975], method="linear")
    return {"ci95_low": float(lower), "ci95_high": float(upper), "valid_days": len(observations)}


# 【核心逻辑123】端点组差先同日配对再进入重采样，两种分组模式使用各自实际端点组。
def endpoint_spread_days(returns: pl.DataFrame, top_group: int) -> pl.DataFrame:
    """输入分组收益表，返回各维度每日最高端点组减最低组(G1)的差值。"""
    high = returns.filter(pl.col("group") == top_group).select(
        "date", "split", "horizon_ns", "sampling", pl.col("mean_return_bps").alias("high_bps"))
    low = returns.filter(pl.col("group") == 1).select(
        "date", "split", "horizon_ns", "sampling", pl.col("mean_return_bps").alias("low_bps"))
    return (high.join(low, on=["date", "split", "horizon_ns", "sampling"], how="inner")
            .with_columns((pl.col("high_bps") - pl.col("low_bps")).alias("daily_spread"))
            .select("date", "split", "horizon_ns", "sampling", "daily_spread"))


# 【核心逻辑122/123】汇总各split×周期×采样的IC与端点组差区间，切分之间互不混用。
def bootstrap_summary(daily_ic: pl.DataFrame, returns: pl.DataFrame,
                      requested_groups: int) -> pl.DataFrame:
    """输入逐日IC与分组收益，返回五个区间字段，序列不足两日的项保持null。"""
    keys = ["split", "horizon_ns", "sampling"]
    ic_series = {tuple(row[key] for key in keys): row["ic"] for row in daily_ic.group_by(keys)
                 .agg(pl.col("ic").drop_nulls().alias("ic")).iter_rows(named=True)}
    spread_series = {tuple(row[key] for key in keys): row["daily_spread"]
                     for row in endpoint_spread_days(returns, requested_groups).group_by(keys)
                     .agg(pl.col("daily_spread").drop_nulls().alias("daily_spread")).iter_rows(named=True)}
    rows = []
    for group in sorted(set(ic_series) | set(spread_series)):
        ic_result = bootstrap_interval(ic_series.get(group, []))
        spread_result = bootstrap_interval(spread_series.get(group, []))
        rows.append({
            "split": group[0], "horizon_ns": group[1], "sampling": group[2],
            "ic_ci95_low": ic_result["ci95_low"], "ic_ci95_high": ic_result["ci95_high"],
            "bootstrap_valid_days": ic_result["valid_days"],
            "group_spread_ci95_low": spread_result["ci95_low"],
            "group_spread_ci95_high": spread_result["ci95_high"],
        })
    schema = {"split": pl.String, "horizon_ns": pl.Int64, "sampling": pl.String,
              "ic_ci95_low": pl.Float64, "ic_ci95_high": pl.Float64, "bootstrap_valid_days": pl.UInt32,
              "group_spread_ci95_low": pl.Float64, "group_spread_ci95_high": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


# 【核心逻辑291】完整五分组才计算单调性，方向沿用训练期IC方向，端点组差按各自分组模式。
def group_quality_statistics(returns: pl.DataFrame, directions: pl.DataFrame,
                             requested_groups: int) -> pl.DataFrame:
    """输入分组收益与训练方向，返回组序单调性、方向一致日比例和端点差为正日比例。"""
    keys = ["split", "horizon_ns", "sampling"]
    direction_map = {(row["horizon_ns"], row["sampling"]): row["train_ic_direction"]
                     for row in directions.iter_rows(named=True)}
    spread_map = {}
    for part in endpoint_spread_days(returns, requested_groups).partition_by(keys):
        spread_map[(part["split"][0], part["horizon_ns"][0], part["sampling"][0])] = part["daily_spread"]
    def spearman(frame: pl.DataFrame) -> float | None:
        """组序号与组均收益的Spearman相关，常数序列返回null。"""
        value = frame.select(pl.corr(pl.col("group").cast(pl.Float64),
                                     pl.col("mean_return_bps"), method="spearman")).item()
        return value if value is not None and math.isfinite(value) else None

    rows = []
    for part in returns.partition_by(keys):
        split, horizon, sampling = part["split"][0], part["horizon_ns"][0], part["sampling"][0]
        direction = direction_map.get((horizon, sampling))
        aggregated = part.group_by("group").agg(pl.col("mean_return_bps").mean()).sort("group")
        monotonicity = None
        consistent_days = valid_days = 0
        if requested_groups == 5 and aggregated.height == 5:
            monotonicity = spearman(aggregated)
            for day in part.partition_by("date"):
                if day.height != 5:
                    continue
                daily = spearman(day)
                if daily is None:
                    continue
                valid_days += 1
                if direction is not None and ((daily > 0 and direction == 1) or (daily < 0 and direction == -1)):
                    consistent_days += 1
        series = spread_map.get((split, horizon, sampling))
        spread = [value for value in (series.to_list() if series is not None else [])
                  if value is not None and math.isfinite(value)]
        positive_days = sum(1 for value in spread if direction is not None and value * direction > 0)
        rows.append({
            "split": split, "horizon_ns": horizon, "sampling": sampling,
            "group_monotonicity": monotonicity,
            "monotonic_days_ratio": (consistent_days / valid_days
                                     if direction is not None and valid_days else None),
            "group_spread_positive_days": (positive_days / len(spread)
                                           if direction is not None and spread else None),
        })
    return pl.DataFrame(rows, schema={"split": pl.String, "horizon_ns": pl.Int64, "sampling": pl.String,
                                      "group_monotonicity": pl.Float64, "monotonic_days_ratio": pl.Float64,
                                      "group_spread_positive_days": pl.Float64})


# 【核心逻辑290】极端分位阈值只在训练期拟合，逐日两端样本均值差按有效日等权。
def extreme_tail_days(table: pl.DataFrame, factor: str, date: str, split: str, horizon_ns: int,
                      sampling: str, p05: float, p95: float) -> dict:
    """输入单日因子标签关联表，返回高于P95与低于P05样本的未来收益均值差。"""
    paired = pl.col('value').is_finite() & pl.col('target').is_finite() & pl.col('label_valid').fill_null(False)
    high = paired & (pl.col('value') > p95)
    low = paired & (pl.col('value') < p05)
    metrics = table.select(
        pl.when(high).then(pl.col('target')).mean().alias('high_bps'),
        pl.when(low).then(pl.col('target')).mean().alias('low_bps'),
        high.sum().alias('high_rows'), low.sum().alias('low_rows'),
    ).row(0, named=True)
    high_bps, low_bps = metrics['high_bps'], metrics['low_bps']
    return {"factor": factor, "date": date,
            "split": split, "horizon_ns": horizon_ns, "sampling": sampling,
            "high_bps": high_bps, "low_bps": low_bps,
            "extreme_spread_bps": (high_bps - low_bps) if high_bps is not None and low_bps is not None else None,
            "high_rows": metrics['high_rows'], "low_rows": metrics['low_rows']}


# 【核心逻辑292】单因子OLS只用训练期估计标准化与斜率，样本外R2以训练均值为基准。
def single_factor_ols(statistics: dict) -> dict:
    """输入各split的(n,Σx,Σx²,Σy,Σy²,Σxy)充分统计量，返回标准化斜率与样本外R2。"""
    train = statistics.get("train")
    result = {"beta_1std_bps": None, "train_rows": None}
    for split in statistics:
        if split != 'train':
            result[f'{split}_rows'] = None
            result[f'{split}_oos_r2'] = None
    if not train or train[0] < 2:
        return result
    count, sum_x, sum_x2, sum_y, sum_y2, sum_xy = train
    mean_x, mean_y = sum_x / count, sum_y / count
    variance_x = sum_x2 / count - mean_x ** 2
    denominator = count * sum_x2 - sum_x ** 2
    if variance_x <= 0 or denominator == 0:
        result["train_rows"] = count
        return result
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = mean_y - slope * mean_x
    result["beta_1std_bps"] = slope * math.sqrt(variance_x)
    result["train_rows"] = count
    for split in [key for key in statistics if key != 'train']:
        values = statistics.get(split)
        if not values or values[0] < 2:
            continue
        n, sx, sx2, sy, sy2, sxy = values
        ss_res = (sy2 - 2 * intercept * sy - 2 * slope * sxy + intercept ** 2 * n
                  + 2 * intercept * slope * sx + slope ** 2 * sx2)
        ss_tot = sy2 - 2 * mean_y * sy + n * mean_y ** 2
        result[f"{split}_rows"] = n
        if ss_tot > 0:
            result[f"{split}_oos_r2"] = 1 - ss_res / ss_tot
    return result
