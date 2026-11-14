"""幅度加权分类输出代表盈利金额占比，不是普通获利概率。"""
import numpy as np
from scipy.special import expit
from mm_research.models.cost_classifier import probability_calibration


# 【非核心】常数特征下，普通校准收敛到胜率，加权校准收敛到盈利金额占比。
def test_weighted_calibration_targets_payoff_share():
    """两笔亏1、一笔赚8：胜率1/3但净收益为正，加权份额应为0.8。"""
    prediction = np.full(3, .5)
    labels = np.array([0., 0., 1.])
    _, ordinary_intercept = probability_calibration(prediction, labels)
    _, weighted_intercept = probability_calibration(prediction, labels, np.array([1., 1., 8.]))
    np.testing.assert_allclose(expit(ordinary_intercept), 1 / 3, atol=1e-4)
    np.testing.assert_allclose(expit(weighted_intercept), .8, atol=1e-4)
