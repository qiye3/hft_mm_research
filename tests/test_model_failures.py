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


# 【非核心】入场秒价格变化不能影响事前状态，质量边界显式保留缺失。
def test_entry_regime_uses_past_only():
    """用可手算价格验证时间偏移，再修改未来部分检查结果不变。"""
    from mm_research.evaluation.model_failures import entry_past_return

    market = np.ones((10, 10))
    market[:, 0] = np.arange(100., 110.)
    rows = np.array([5, 7])
    expected = np.log(np.array([104. / 102., 106. / 104.])) * 10000
    np.testing.assert_allclose(entry_past_return(market, rows, 2), expected)
    market[7:, 0] = 999.
    np.testing.assert_allclose(entry_past_return(market, rows, 2), expected)
    market[6, 9] = 2.
    assert np.isnan(entry_past_return(market, rows, 2)[1])
