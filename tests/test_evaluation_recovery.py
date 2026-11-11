"""评估末端空值类型、有限内存Bootstrap及发布检查点回归。"""

import datetime as dt
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from mm_research.data.storage import write_parquet
from mm_research.evaluation.metrics import bootstrap_interval
from mm_research.evaluation.report import evaluate_factor_batch, publish_factor_evaluation


# 【非核心】分批生成重采样索引必须保持原2000次整块计算的结果。
def test_chunked_bootstrap_matches_reference():
    """使用318个有效日，覆盖真实失败时的重采样数组尺寸。"""
    observations = np.sin(np.arange(318, dtype=float))
    generator = np.random.default_rng(20261108)
    draws = generator.integers(0, len(observations), size=(2000, len(observations)))
    expected = np.quantile(observations[draws].mean(axis=1), [0.025, 0.975])
    actual = bootstrap_interval(observations.tolist())
    np.testing.assert_array_equal([actual['ci95_low'], actual['ci95_high']], expected)


# 【非核心】前110条极端收益为空时，后续Float64仍应正常发布。
def test_publish_extremes_after_long_null_prefix(tmp_path):
    """直接走真实汇总入口，隔离绘图，检查极端收益三列的正式类型。"""
    correlations = []
    distributions = []
    groups = []
    extremes = []
    for index in range(120):
        date = (dt.date(2025, 1, 1) + dt.timedelta(days=index)).isoformat()
        split = 'train' if index < 60 else 'val_select'
        keys = {'factor': 'sample', 'date': date, 'split': split,
                'horizon_ns': 180_000_000_000, 'sampling': 'all'}
        correlations.append({**keys, 'ic': 0.1, 'rank_ic': 0.1, 'paired_rows': 100})
        distributions.append({'factor': 'sample', 'date': date, 'split': split,
                              'rows': 100, 'valid_rows': 100})
        groups.append(pl.DataFrame([
            {**keys, 'group': group, 'samples': 20, 'factor_mean': float(group),
             'mean_return_bps': group * 0.001, 'actual_groups': 5, 'requested_groups': 5}
            for group in range(1, 6)
        ]))
        high = 0.044684 if index >= 110 else None
        low = -0.012 if index >= 110 else None
        extremes.append({**keys, 'high_bps': high, 'low_bps': low,
                         'extreme_spread_bps': high - low if high is not None else None,
                         'high_rows': 1 if high is not None else 0,
                         'low_rows': 1 if low is not None else 0})
    state = {
        'correlations': correlations, 'distributions': distributions, 'groups': groups,
        'extremes': extremes, 'boundaries': {'mode': 'quantile', 'cutoffs': [1, 2, 3, 4]},
        'identity': {'test': 'long_null_prefix'}, 'output': tmp_path,
        'incremental': [], 'ols': {}, 'histogram': np.zeros(80), 'edges': np.arange(81),
    }
    with patch('mm_research.evaluation.report.render_factor_report'):
        publish_factor_evaluation('sample', {'groups': 5}, state)
    frame = pl.read_parquet(tmp_path / 'extreme_returns.parquet')
    assert frame['high_bps'].count() == 10
    assert frame.schema['high_bps'] == pl.Float64
    assert frame.schema['low_bps'] == pl.Float64
    assert frame.schema['extreme_spread_bps'] == pl.Float64


# 【非核心】发布失败后恢复已算日级统计，不再次扫描因子和标签。
def test_publish_checkpoint_avoids_replay(tmp_path):
    """第一次人为中断发布，第二次禁止行情计算仍须完成。"""
    date = '2026-07-12'
    name = 'sample_mean_10s'
    dataset = {'dates': [date], 'symbol': 'BTCUSDT'}
    start = int(dt.datetime(2026, 7, 12, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    keys = {'symbol': ['BTCUSDT'] * 100, 'date': [date] * 100,
            'timestamp_ns': start + np.arange(100, dtype=np.int64) * 1_000_000_000}
    factor = pl.DataFrame({**keys, name: np.arange(100, dtype=float)})
    write_parquet(factor.lazy(), tmp_path / f'factors/test/{name}.parquet',
                  {'kind': 'factor', 'dataset': dataset, 'definition': {'dates': [date]}})
    labels = pl.DataFrame({**keys, 'label_valid': [True] * 100,
                           'label_end_timestamp_ns': keys['timestamp_ns'] + 1_000_000_000,
                           'y_mid_1s_bps': np.arange(100, dtype=float) * 2})
    write_parquet(labels.lazy(), tmp_path / f'labels/test/horizon=1s/{date}.parquet',
                  {'kind': 'labels', 'dataset': dataset, 'horizon_ns': 1_000_000_000})
    config = {
        'factor_root': 'factors', 'factor_version': 'test', 'label_root': 'labels',
        'label_version': 'test', 'dates': [date], 'factors': [name], 'output': 'results',
        'train_end': date, 'validation_dev_end': date, 'validation_end': date,
        'purge_ns': 300_000_000_000, 'factor_lag_ns': 0, 'groups': 5,
        'horizons_ns': [1_000_000_000], 'output_interval_ns': 1_000_000_000,
        'ols_factors': [], 'workers': 1, 'factor_batch_size': 12,
    }
    task = (tmp_path, config, [name])
    with patch('mm_research.evaluation.report.publish_factor_evaluation', side_effect=RuntimeError('publish interrupted')):
        with pytest.raises(RuntimeError, match='publish interrupted'):
            evaluate_factor_batch(task)
    checkpoint = tmp_path / f'results/.pending/{name}/state.pkl'
    assert checkpoint.exists()
    with patch('mm_research.evaluation.report.fit_group_boundaries', side_effect=AssertionError('rescan')):
        with patch('mm_research.evaluation.report.load_evaluation_targets', side_effect=AssertionError('replay')):
            with patch('mm_research.evaluation.report.render_factor_report'):
                results = evaluate_factor_batch(task)
    assert not checkpoint.exists()
    assert results == [{'factor': name, 'cached': False}]


# 【非核心】只恢复部分窗口时，已完成的长窗口仍须参与共同样本筛选。
def test_cached_window_keeps_common_sample_mask(tmp_path):
    """保留600秒窗口缓存，其他窗口重算后共同样本量必须不变。"""
    import shutil

    date = '2026-07-12'
    names = [f'sample_mean_{window}s' for window in [10, 60, 180, 600]]
    dataset = {'dates': [date], 'symbol': 'BTCUSDT'}
    start = int(dt.datetime(2026, 7, 12, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    keys = {'symbol': ['BTCUSDT'] * 100, 'date': [date] * 100,
            'timestamp_ns': start + np.arange(100, dtype=np.int64) * 1_000_000_000}
    for index, name in enumerate(names):
        values = np.arange(100, dtype=float)
        values[:(index + 1) * 5] = np.nan
        factor = pl.DataFrame({**keys, name: values})
        write_parquet(factor.lazy(), tmp_path / f'factors/test/{name}.parquet',
                      {'kind': 'factor', 'dataset': dataset, 'definition': {'dates': [date]}})
    labels = pl.DataFrame({**keys, 'label_valid': [True] * 100,
                           'label_end_timestamp_ns': keys['timestamp_ns'] + 1_000_000_000,
                           'y_mid_1s_bps': np.arange(100, dtype=float) * 2})
    write_parquet(labels.lazy(), tmp_path / f'labels/test/horizon=1s/{date}.parquet',
                  {'kind': 'labels', 'dataset': dataset, 'horizon_ns': 1_000_000_000})
    config = {
        'factor_root': 'factors', 'factor_version': 'test', 'label_root': 'labels',
        'label_version': 'test', 'dates': [date], 'factors': names, 'output': 'results',
        'train_end': date, 'validation_dev_end': date, 'validation_end': date,
        'purge_ns': 300_000_000_000, 'factor_lag_ns': 0, 'groups': 5,
        'horizons_ns': [1_000_000_000], 'output_interval_ns': 1_000_000_000,
        'ols_factors': [], 'workers': 1, 'factor_batch_size': 12,
    }
    with patch('mm_research.evaluation.report.render_factor_report'):
        evaluate_factor_batch((tmp_path, config, names))
    for name in names[:-1]:
        shutil.rmtree(tmp_path / 'results' / name)
    with patch('mm_research.evaluation.report.render_factor_report'):
        evaluate_factor_batch((tmp_path, config, names))
    for name in names:
        daily = pl.read_parquet(tmp_path / 'results' / name / 'daily_ic.parquet')
        assert daily.filter(pl.col('sampling') == 'common')['paired_rows'].item() == 80
