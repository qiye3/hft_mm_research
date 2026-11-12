"""状态条件价值仅使用已知状态和冻结参数，未知状态不能开仓。"""

import numpy as np

from mm_research.models.state_value import volatility_states, conditional_values


# 【非核心】收益幅度随状态变化，但相同冻结概率不读取未来收益。
def test_conditional_values_respect_state_and_missing():
    """50%胜率在低波动下不够付费，在高波动下按明确幅度计算价值。"""
    probability = np.array([.5, .5, .5, .5])
    states = volatility_states(np.array([1., 3., 8., np.nan]), np.array([2., 5.]))
    definitions = [
        {'win_mean_bps': 2., 'loss_mean_bps': -8., 'slope': 1., 'intercept': 0.},
        {'win_mean_bps': 10., 'loss_mean_bps': -10., 'slope': 1., 'intercept': 0.},
        {'win_mean_bps': 30., 'loss_mean_bps': -10., 'slope': 1., 'intercept': 0.},
    ]
    values = conditional_values(probability, states, definitions, False)
    np.testing.assert_allclose(values, [-3., 0., 10., np.nan], equal_nan=True)
    np.testing.assert_allclose(values, conditional_values(probability, states, definitions, True), equal_nan=True)
    changed = probability.copy()
    changed[2] = .9
    np.testing.assert_array_equal(values[:2], conditional_values(changed, states, definitions, True)[:2])
