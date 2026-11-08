"""看板十二组条件收益；不修改或替代主项目五组评估产物。"""
from __future__ import annotations

import numpy as np
import polars as pl

GROUP_NAMES = ['G0.1', *[f'G{index}' for index in range(1, 11)], 'G9.9']
QUANTILES = [.01, *[index / 10 for index in range(1, 10)], .99]
SCOPES = {
    'full': ('2026-07-01', '2026-07-30', '全期研究'),
    'development': ('2026-07-01', '2026-07-18', '开发期回溯切分'),
    'validation': ('2026-07-19', '2026-07-24', '验证期回溯切分'),
    'test': ('2026-07-25', '2026-07-30', '测试期回溯演练（日期已看过）'),
}


# 【非核心】常数、样本不足的相关性保持不可计算，不返回伪造的零系数。
def pearson(values: np.ndarray, targets: np.ndarray) -> float | None:
    """输入共同有限的两个数组，输出Pearson相关性或null。"""
    if len(values) < 2 or np.ptp(values) == 0 or np.ptp(targets) == 0:
        return None
    return float(np.corrcoef(values, targets)[0, 1])


# 【非核心】十组互斥、两尾重叠；相同数值不按时间顺序强拆。
def group_masks(values: np.ndarray, thresholds: list[float]) -> list[np.ndarray]:
    """输入因子值和1/10/…/90/99%边界，返回十二个固定名称对应掩码。"""
    bins = np.searchsorted(thresholds[1:-1], values, side='left')
    return [values <= thresholds[0], *[bins == index for index in range(10)],
            values > thresholds[-1]]


# 【非核心】分位数仅来自因子，不使用未来标签有效性决定阈值。
def group_statistics(frame: pl.DataFrame, thresholds: list[float]) -> list[dict]:
    """输入单日因子标签关联表，输出十二组样本数、条件收益与组内IC。"""
    paired = frame.filter(pl.col('value').is_finite() & pl.col('target').is_finite()
                          & pl.col('label_valid'))
    values = paired['value'].to_numpy()
    targets = paired['target'].to_numpy()
    rows = []
    for name, mask in zip(GROUP_NAMES, group_masks(values, thresholds)):
        count = int(mask.sum())
        mean_return = None
        if count:
            mean_return = float(targets[mask].mean())
        rows.append({
            'group': name, 'count': count, 'bps': mean_return,
            'ic': pearson(values[mask], targets[mask]),
        })
    return rows


# 【非核心】使用连续三日移动块重采样，保留日内标签重叠及部分跨日依赖。
def confidence_interval(values: list[float | None]) -> list[float] | None:
    """输入按日期排列的日指标，返回固定种子2000次bootstrap均值95%区间。"""
    observations = np.asarray([value for value in values if value is not None], dtype=float)
    count = len(observations)
    if count < 3:
        return None
    generator = np.random.default_rng(20261107)
    starts = generator.integers(0, count - 2, size=(2000, (count + 2) // 3))
    indices = (starts[..., None] + np.arange(3)).reshape(2000, -1)[:, :count]
    means = observations[indices].mean(axis=1)
    return np.quantile(means, [.025, .975], method='linear').tolist()


# 【非核心】日等权聚合保留空组和尾部重叠的语义，样本数不跨十二组相加。
def aggregate_groups(days: list[dict], mode: str, horizon: int) -> list[dict]:
    """输入选定日期的统计，输出逐组日均收益、日均IC与平均每日样本量。"""
    output = []
    for index, name in enumerate(GROUP_NAMES):
        rows = [day['groups'][mode][str(horizon)][index] for day in days]
        returns = [row['bps'] for row in rows if row['bps'] is not None]
        correlations = [row['ic'] for row in rows if row['ic'] is not None]
        output.append({
            'group': name,
            'bps': float(np.mean(returns)) if returns else None,
            'ic': float(np.mean(correlations)) if correlations else None,
            'count': sum(row['count'] for row in rows),
            'average_count': sum(row['count'] for row in rows) / len(days),
            'valid_days': len(returns),
        })
    return output
