"""短周期方向确认不读取标签，拒绝错位时间网格。"""
from pathlib import Path
import numpy as np
import polars as pl
from mm_research.strategies.account_sizing import directional_entry_filter


# 【非核心】正负、零、缺失预测及时间键错位都用明确样本覆盖。
def test_direction_filter(tmp_path: Path):
    """只有同向有限预测准入；读取列不包含未来标签。"""
    filename = tmp_path / 'prediction.parquet'
    frame = pl.DataFrame({
        'timestamp_ns': np.arange(5, dtype='int64') * 1_000_000_000,
        'prediction_log_bps': [1., -1., 0., float('nan'), 1.],
    })
    frame.write_parquet(filename)
    signal = np.array([1., -1., 1., 1., -1.])
    np.testing.assert_array_equal(directional_entry_filter(filename, signal, 0), [True, True, False, False, False])
    frame.with_columns(pl.col('timestamp_ns') + 1).write_parquet(filename)
    try:
        directional_entry_filter(filename, signal, 0)
    except ValueError:
        pass
    else:
        raise AssertionError('错位时间网格必须拒绝')
