"""生成看板独立诊断快照，完整发布后切换current指针。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import polars as pl

from .dataset import (
    FACTORS, FEATURES, PROJECT, RESULTS, SHORT_NAMES, correlation_days,
    delay_diagnostics, evaluate_day, metadata, source_locks,
)
from .statistics import QUANTILES, SCOPES

PREVIEW = PROJECT / 'workspace/results/factorboard'


# 【非核心】只在看板自己的输出目录写JSON，不复制原始因子和标签。
def write_json(path: Path, payload: dict | list) -> None:
    """严格输出有限统计，发布文件使用原子替换。"""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf8')
    temporary.replace(path)


# 【非核心】每日统一边界的直方图，所有分位统计采用线性插值。
def histogram(values: pl.DataFrame, name: str, dates: list[str]) -> dict:
    """直接统计既有因子有限值，输出全期36桶及逐日密度。"""
    finite = values.filter(pl.col(name).is_finite())
    counts, edges = np.histogram(finite[name].to_numpy(), bins=36)
    daily = []
    for date in dates:
        day_counts, _ = np.histogram(finite.filter(pl.col('date') == date)[name].to_numpy(), bins=edges)
        daily.append({'day': date, 'counts': day_counts.tolist()})
    return {'edges': edges.tolist(), 'counts': counts.tolist(), 'daily': daily}


# 【非核心】固定阈值仅用开发期因子值，保留日级统计供不同范围重复读取。
def prepare_factor(name: str, original_ic: pl.DataFrame) -> dict:
    """构造一份因子的十二组、覆盖、多周期、延迟和分布快照。"""
    path = FACTORS / f'{name}.parquet'
    with source_locks([path]):
        identity = metadata(path)
        values = pl.read_parquet(path)
        dates = identity['definition']['dates']
        development = values.filter(pl.col('date') <= SCOPES['development'][1]).filter(pl.col(name).is_finite())
        fixed = [development[name].quantile(quantile, interpolation='linear') for quantile in QUANTILES]
        daily = []
        delays = []
        for date in dates:
            day = values.filter(pl.col('date') == date).rename({name: 'value'})
            daily.append(evaluate_day(day, date, fixed, original_ic, identity['dataset']))
            delays.extend(delay_diagnostics(day, date))
        distribution = histogram(values, name, dates)
    return {
        'id': name, 'short_name': SHORT_NAMES[name], 'metadata': identity,
        'fixed_thresholds': fixed, 'daily': daily, 'delays': delays,
        'histogram': distribution,
    }


# 【非核心】整个快照写完再发布指针，失败时旧看板继续读取原快照。
def prepare() -> None:
    """只读当前研究结果，生成看板专属十二组数据和全日期相关性。"""
    original_ic = pl.read_parquet(RESULTS / 'daily_ic.parquet')
    names = sorted(original_ic['factor'].unique())
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    runs = PREVIEW / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.preparing-', dir=runs) as temporary:
        staging = Path(temporary)
        for name in names:
            payload = prepare_factor(name, original_ic.filter(pl.col('factor') == name))
            write_json(staging / f'{name}.json', payload)
            print(f'十二组与延迟诊断完成 {name}', flush=True)
        dates = payload['metadata']['definition']['dates']
        correlations = correlation_days(names, dates)
        write_json(staging / 'correlations.json', correlations)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
        manifest = {
            'run_id': run_id, 'completed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'factor_version': FACTORS.name, 'feature_version': FEATURES.name,
            'label_version': 'market_mid_columns_v1', 'factor_ids': names, 'dates': dates,
            'code_commit': commit, 'code_state': '工作区未提交；看板独立适配',
            'source': '既有单因子Parquet及核心daily_ic；仅看板十二组扩展',
            'dataset': payload['metadata']['dataset'],
            'scopes': {key: list(value) for key, value in SCOPES.items()},
            'group_spec': 'G1-G10十分位；G0.1≤P1；G9.9>P99；两尾与端点重叠',
            'bootstrap': {'method': '连续三日移动块', 'draws': 2000, 'seed': 20261107},
            'cost_defaults': {'fee_per_side_bps': 5, 'slippage_per_side_bps': 0, 'spread_roundtrip_bps': 0},
            'cost_source': '项目基线配置taker费率0.0005换算；仅情景假设，不代表账户历史费率',
        }
        write_json(staging / 'manifest.json', manifest)
        final = runs / run_id
        staging.rename(final)
    write_json(PREVIEW / 'current.json', {'run_id': run_id})
    print(f'发布完成 {run_id}', flush=True)
