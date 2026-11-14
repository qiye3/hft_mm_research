"""诊断指标区分全部预测与有效标签，不将无定义相关填零。"""

import numpy as np
from mm_research.evaluation.model_failures import finite_correlation, output_profile


# 【非核心】预测正值占比不按未来标签过滤，获利率只在有效未来标签上定义。
def test_profile_label_population():
    """第三行标签缺失不能从模型输出分布中消失。"""
    edge = np.array([-1., 2., 3., 4.])
    result = output_profile(edge, np.array([.2, .6, .7, .8]), np.array([-2., 1., np.nan, -3.]))
    assert result['positive_edge_fraction'] == .75
    assert result['positive_edge_win_rate'] == .5
    assert result['grid_rows'] == 4
    assert result['label_rows'] == 3
    assert finite_correlation(np.ones(4), edge) is None
