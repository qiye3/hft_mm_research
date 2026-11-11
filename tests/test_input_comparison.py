"""新输入时间对齐、拟合前缀标准化和独立缺失通道。"""

import numpy as np
import polars as pl
import json

from mm_research.models.input_comparison import materialize_new_column, select_new_inputs
from mm_research.multihorizon.common import DAY, SECOND


# 【非核心】未来极端值和缺失值不得改变拟合前缀的归一化参数。
def test_new_column_uses_previous_second_and_fit_only(tmp_path):
    """逐行对照上一秒输入、早停位置、验证位置及缺失标记，窗口在午夜连续。"""
    filename = tmp_path / 'factor.parquet'
    start = 86398 * SECOND
    raw = [1.0, 3.0, None, 7.0, 100000.0, 2.0]
    pl.DataFrame({'timestamp_ns': start + np.arange(len(raw)) * SECOND, 'factor': raw}).write_parquet(filename)
    rows = {'fit': np.array([1, 2]), 'tune': np.array([3, 4]), 'val_dev': np.array([5])}
    caches = {name: np.empty((len(positions), 2), dtype='float32') for name, positions in rows.items()}
    result = materialize_new_column((0, {'factor': 'factor', 'source_file': str(filename)}, rows, caches, start, len(raw)))
    assert result['mean'] == 2.0
    assert result['scale'] == 1.0
    np.testing.assert_array_equal(caches['fit'], [[-1, 0], [1, 0]])
    np.testing.assert_array_equal(caches['tune'], [[0, 1], [5, 0]])
    np.testing.assert_array_equal(caches['val_dev'], [[20, 0]])


# 【非核心】选窗截止日后的强相关和旧同名列不得影响本轮新输入排序。
def test_selection_excludes_tune_days_and_old_names(tmp_path):
    """内部早停尾部刻意反转候选优劣，选列仍由前两天拟合数据决定。"""
    source = tmp_path / 'source'
    (source / 'e0').mkdir(parents=True)
    fold = source / 'e1/folds/final'
    fold.mkdir(parents=True)
    np.save(fold / 'fit_rows.npy', np.array([1, DAY, 2 * DAY + 100]))
    (source / 'e0/manifest.json').write_text(json.dumps({
        'dates': ['2025-05-09'], 'feature_names': ['old_10s'],
    }))
    results = tmp_path / 'results'
    (results / 'summary').mkdir(parents=True)
    names = ['alpha_10s', 'alpha_60s', 'beta_10s', 'old_10s']
    records = []
    for name, early, late in zip(names, [0.2, 0.1, 0.05, 0.9], [-0.1, 0.2, 0.99, 0.99]):
        folder = results / 'evaluation/gen001' / name
        folder.mkdir(parents=True)
        pl.DataFrame({
            'factor': [name] * 3, 'date': ['2025-05-09', '2025-05-10', '2025-05-11'],
            'horizon_ns': [180 * SECOND] * 3, 'sampling': ['common'] * 3,
            'rank_ic': [early, early, late],
        }).write_parquet(folder / 'daily_ic.parquet')
        records.append({'batch': 'gen001', 'factor': name, 'source_file': str(folder / 'factor.parquet')})
    pl.DataFrame(records).write_csv(results / 'summary/factor_inventory.csv')
    selected = select_new_inputs({'factor_results': str(results), 'new_features': 1}, source, 180)
    assert selected['factor'].to_list() == ['alpha_10s']
    assert selected['selection_before_utc_date'].to_list() == ['2025-05-11']
