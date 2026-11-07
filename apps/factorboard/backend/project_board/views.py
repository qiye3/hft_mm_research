"""从紧凑日统计构造不同范围的只读看板响应。"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .prepare import PREVIEW
from .statistics import SCOPES, aggregate_groups, confidence_interval


# 【非核心】每次请求先读取原子指针，缓存键包含已完成运行ID。
def current_run() -> tuple[Path, dict]:
    """返回当前完整看板结果目录和身份，不混合两个运行的文件。"""
    run_id = json.loads((PREVIEW / 'current.json').read_text())['run_id']
    root = PREVIEW / 'runs' / run_id
    return root, read_json(str(root / 'manifest.json'))


# 【非核心】快照发布后不可变，按运行文件路径缓存紧凑统计。
@lru_cache(maxsize=64)
def read_json(path: str) -> dict | list:
    """读取一个已完成JSON；缓存不按因子名跨运行复用。"""
    return json.loads(Path(path).read_text())


# 【非核心】空指标保留null，避免将不可计算值当作零。
def mean(values: list[float | None]) -> float | None:
    """对明确可用的日指标等权汇总。"""
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(np.mean(valid))


# 【非核心】统计范围只过滤日期，不重新拟合开发期阈值或因子方向。
def evaluation(root: Path, name: str, scope: str, mode: str, horizon: int) -> dict:
    """返回覆盖、原始IC、十二组、区间、衰减和分布的选定范围视图。"""
    payload = read_json(str(root / f'{name}.json'))
    start, end, title = SCOPES[scope]
    days = [day for day in payload['daily'] if start <= day['date'] <= end]
    horizon_rows = []
    for period in [100, 500, 1000, 5000]:
        observations = [row for day in days for row in day['ic'] if row['h'] == period]
        horizon_rows.append({
            'h': period, 'ic': mean([row['ic'] for row in observations]),
            'rank_ic': mean([row['rank_ic'] for row in observations]),
        })
    selected_ic = [row for day in days for row in day['ic'] if row['h'] == horizon]
    ic_values = [row['ic'] for row in selected_ic]
    valid_ic = [value for value in ic_values if value is not None]
    standard_deviation = float(np.std(valid_ic, ddof=1)) if len(valid_ic) > 1 else 0
    coverages = [day['coverage'][str(horizon)] for day in days]
    coverage = {key: sum(row[key] for row in coverages) for key in coverages[0]}
    interval = payload['metadata']['dataset']['interval_ns']
    coverage['expected'] = len(days) * (86_400_000_000_000 // interval)
    coverage['missing_grid'] = coverage['expected'] - coverage['rows']
    grouped = aggregate_groups(days, mode, horizon)
    daily_spreads = []
    for day in days:
        rows = day['groups'][mode][str(horizon)]
        low = rows[1]['bps']
        high = rows[10]['bps']
        spread = None
        if low is not None and high is not None:
            spread = high - low
        daily_spreads.append({'day': day['date'], 'spread': spread})
    selected_dates = {day['date'] for day in days}
    histogram_days = [row for row in payload['histogram']['daily'] if row['day'] in selected_dates]
    counts = np.sum([row['counts'] for row in histogram_days], axis=0)
    edges = payload['histogram']['edges']
    hist = [{'x': (edges[index] + edges[index + 1]) / 2, 'v': float(count / counts.sum())}
            for index, count in enumerate(counts)]
    delays = []
    for delay in [0, 100, 200, 500, 1000]:
        rows = [row for row in payload['delays'] if row['day'] in selected_dates and row['delay_ms'] == delay]
        delays.append({'label': f'{delay}ms', 'ic': mean([row['ic'] for row in rows]),
                       'count': sum(row['count'] for row in rows)})
    return {
        'id': name, 'short_name': payload['short_name'], 'scope': scope, 'scope_title': title,
        'dates': sorted(selected_dates), 'horizon': horizon, 'mode': mode,
        'metadata': payload['metadata'], 'coverage': coverage, 'groups': grouped,
        'ic': mean(ic_values), 'rank_ic': mean([row['rank_ic'] for row in selected_ic]),
        'icir': mean(ic_values) / standard_deviation if standard_deviation > 0 else None,
        'ic_ci': confidence_interval(ic_values),
        'positive': sum(value > 0 for value in valid_ic) / len(valid_ic) if valid_ic else None,
        'daily_ic': [{'day': day['date'], **next(row for row in day['ic'] if row['h'] == horizon)} for day in days],
        'horizons': horizon_rows, 'delays': delays, 'daily_spreads': daily_spreads,
        'spread_ci': confidence_interval([row['spread'] for row in daily_spreads]),
        'spread': mean([row['spread'] for row in daily_spreads]),
        'hist': hist, 'pct': [{'day': day['date'], **day['percentiles']} for day in days],
        'densities': [(np.array(row['counts']) / sum(row['counts'])).tolist() for row in histogram_days],
        'fixed_thresholds': payload['fixed_thresholds'],
        'daily_group_ic': [{'day': day['date'], 'groups': day['groups'][mode][str(horizon)]} for day in days],
    }


# 【非核心】Pearson按全期充分统计量精确合并；Rank显示日均而非伪称全期排名。
def correlations(root: Path, names: list[str], scope: str, method: str) -> dict:
    """返回同一范围、六因子共同样本的矩阵及各日相关。"""
    start, end, _ = SCOPES[scope]
    source = read_json(str(root / 'correlations.json'))
    selected = [row for row in source if start <= row['date'] <= end]
    matrix = []
    for left in names:
        matrix_row = []
        for right in names:
            rows = [row for row in selected if row['left'] == left and row['right'] == right]
            if method == 'spearman':
                coefficient = mean([row['spearman'] for row in rows])
            else:
                totals = {key: sum(row[key] for row in rows)
                          for key in ['count', 'sum_x', 'sum_y', 'sum_x2', 'sum_y2', 'sum_xy']}
                count = totals['count']
                covariance = totals['sum_xy'] - totals['sum_x'] * totals['sum_y'] / count
                variance_x = totals['sum_x2'] - totals['sum_x'] ** 2 / count
                variance_y = totals['sum_y2'] - totals['sum_y'] ** 2 / count
                coefficient = float(np.clip(covariance / np.sqrt(variance_x * variance_y), -1, 1))
            matrix_row.append(coefficient)
        matrix.append(matrix_row)
    dates = sorted({row['date'] for row in selected})
    sample_rows = sum(row['count'] for row in selected if row['left'] == names[0] and row['right'] == names[0])
    return {'ids': names, 'values': matrix, 'dates': dates, 'count': sample_rows,
            'method': method, 'mask': '六因子共同有限样本',
            'daily': [row for row in selected if row['left'] in names and row['right'] in names]}
