"""宽特征费用模型复用输入缓存，不改变原有列顺序。"""
from pathlib import Path
import numpy as np
from mm_research.models.intraday import input_parts


# 【非核心】原三组和可选追加组按固定次序读取，缺失通道留在各组内部。
def test_explicit_extra_inputs(tmp_path: Path):
    """不同数值标记四组缓存，检查显式追加及原路径行为。"""
    macro = tmp_path / 'macro'
    folders = [tmp_path / 'old', tmp_path / 'spot', macro / 'all/inputs/180s', tmp_path / 'extra']
    for index, folder in enumerate(folders):
        folder.mkdir(parents=True)
        np.save(folder / 'fit_x.npy', np.full((3, 2), index, dtype='float32'))
    config = {'local_cache': str(folders[0]), 'spot_inputs': str(folders[1])}
    original = input_parts(config, macro, 'fit')
    assert len(original) == 3
    config['extra_inputs'] = str(folders[3])
    expanded = input_parts(config, macro, 'fit')
    assert len(expanded) == 4
    for index, matrix in enumerate(expanded):
        np.testing.assert_array_equal(matrix, np.full((3, 2), index))
