"""因子去冗余的成对缺失相关定义检查。"""

import numpy as np
from mm_research.models.feature_pool import pairwise_correlation


# 【非核心】每对列使用各自共同有效行，不能将缺失值填零后直接求相关。
def test_pairwise_missing_values():
    """独立numpy参考覆盖正相关、负相关与不同缺失位置。"""
    values = np.array([[1., 3., 8.], [2., np.nan, 7.], [3., 5., 6.], [4., 8., np.nan], [5., 7., 4.]])
    result = pairwise_correlation(values, 3)
    for left in range(3):
        for right in range(3):
            valid = np.isfinite(values[:, left]) & np.isfinite(values[:, right])
            expected = np.corrcoef(values[valid, left], values[valid, right])[0, 1]
            np.testing.assert_allclose(result[left, right], expected, atol=1e-12)


# 【非核心】选列只影响新增输入，保留旧列以及新增值与缺失通道的对应顺序。
def test_selected_input_cache(tmp_path):
    """临时矩阵明确区分列编号，验证组合矩阵和原始缓存互不覆盖。"""
    import json
    from mm_research.models.input_comparison import load_input_group

    old = tmp_path / 'old'
    new = tmp_path / 'new'
    old.mkdir()
    new.mkdir()
    old_values = np.arange(8, dtype='float32').reshape(4, 2)
    new_values = np.arange(24, dtype='float32').reshape(4, 6)
    np.save(old / 'fit_x.npy', old_values)
    np.save(new / 'fit_x.npy', new_values)
    selection = tmp_path / 'selected.json'
    selection.write_text(json.dumps({'matrix_columns': [2, 0, 5, 3]}))
    config = {'local_cache': str(old), 'input_selection': {'source': str(new), 'file': str(selection)}}
    actual = load_input_group(config, tmp_path, 180, 'combined', 'fit')
    np.testing.assert_array_equal(actual, np.concatenate([old_values, new_values[:, [2, 0, 5, 3]]], axis=1))
    np.testing.assert_array_equal(np.load(new / 'fit_x.npy'), new_values)
