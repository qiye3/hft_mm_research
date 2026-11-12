"""持仓管理沿用交易账本，验证可见时钟、部分退出和既有规则。"""

import numpy as np

from mm_research.multihorizon.inventory import simulate_inventory


# 【非核心】固定可执行深度与决策盘口分离，便于测试到达价格不能触发提前退出。
def replay(policy, signals, books, management, minimum_hold=5, maximum_hold=30):
    """返回共用模拟器的账本与成交；测试只改变策略管理分支。"""
    empty = np.array([], dtype=float)
    return simulate_inventory(
        signals[:, None], books, np.arange(len(signals), dtype=float), np.ones(1),
        policy, 0, minimum_hold, 1., 0., 1., .002, .0005, 100., .001,
        maximum_hold, 10000., empty, empty, empty, management,
    )


# 【非核心】合成盘口保留十档和实际数量步长，不绕开共用成交逻辑。
def inputs():
    """创建足够末端平仓的网格，首秒发出唯一入场信号。"""
    books = np.ones((100, 40))
    books[:, :10] = 100000. - np.arange(10) * .1
    books[:, 20:30] = 100000.1 + np.arange(10) * .1
    signals = np.zeros(100)
    signals[0] = 1.
    management = np.column_stack([books[:, 0], books[:, 20], np.full(100, 10.), np.ones(100)])
    return signals, books, management


# 【非核心】新管理参数不能改变固定持仓分支，旧规则仍在指定秒平仓。
def test_fixed_policy_unchanged():
    """可选管理输入传入与否产生完全相同的旧策略账本。"""
    signals, books, management = inputs()
    plain = replay(0, signals, books, None)
    with_management = replay(0, signals, books, management)
    np.testing.assert_array_equal(plain[0], with_management[0])
    np.testing.assert_array_equal(plain[1][:, 0], [0., 5.])


# 【非核心】退出请求在部分成交后锁定，后续信号恢复也必须先平掉残仓。
def test_signal_exit_latches_partial_fill():
    """第五秒仅成交半仓，第六秒信号恢复仍完成剩余平仓。"""
    signals, books, management = inputs()
    signals[6] = 1.
    books[5, 10:20] = 0.
    books[5, 10] = .001
    _, fills, position = replay(4, signals, books, management)
    np.testing.assert_array_equal(fills[:, 0], [0., 5., 6.])
    np.testing.assert_allclose(fills[:, 2], [.002, .001, .001])
    assert position == 0.


# 【非核心】到达时盘口的价格下跌不能作为尚未收到的止损信号。
def test_stop_uses_visible_quotes_only():
    """到达盘口先跌，决策盘口下一秒才跌，退出也必须等到下一秒。"""
    signals, books, management = inputs()
    books[2:, :10] -= 1000.
    books[2:, 20:30] -= 1000.
    management[3:, :2] -= 1000.
    _, fills, _ = replay(5, signals, books, management)
    np.testing.assert_array_equal(fills[:, 0], [0., 3.])


# 【非核心】波动率缺失的管理策略保持空仓，避免隐式套用任意风险尺度。
def test_missing_risk_does_not_enter():
    """没有完整事前窗口时不生成波动保护策略的入场。"""
    signals, books, management = inputs()
    management[:, 2] = np.nan
    _, fills, position = replay(5, signals, books, management)
    assert len(fills) == 0
    assert position == 0.


# 【非核心】线性滑窗与已有逐窗参考一致，并且未来价格变化不影响当前风险尺度。
def test_visible_volatility_matches_reference():
    """包含缺价和质量边界，跨日连续取前缀，检查严格早于当前秒。"""
    from mm_research.strategies.stable_strategy import visible_management
    from mm_research.strategies.cost_attribution import past_volatility

    market = np.ones((2200, 10))
    market[:, 0] = 100000. + np.sin(np.arange(2200) / 10.)
    market[:, 1] = market[:, 0] - .1
    market[:, 2] = market[:, 0] + .1
    market[900, 0] = np.nan
    market[1600:, 9] = 2.
    rows = np.arange(700, 2100)
    actual = visible_management(market, 700, 2100)[:, 2]
    expected = past_volatility(market, rows)
    np.testing.assert_allclose(actual, expected, atol=1e-8, equal_nan=True)
    market[1800:, :3] *= 2.
    changed = visible_management(market, 700, 2100)[:, 2]
    np.testing.assert_allclose(actual[:1101], changed[:1101], equal_nan=True)


# 【非核心】单模型极端看多不应自动变成三个模型的共同交易机会。
def test_combination_handles_disagreement():
    """中位数和分歧惩罚压低单一极值，一致认可严格排除非正价值。"""
    from mm_research.strategies.stable_strategy import combine_values

    edges = np.array([[[100., -1.]], [[-2., -1.]], [[-2., -1.]]])
    assert combine_values(edges, 'mean')[0, 0] == 32.
    assert combine_values(edges, 'median')[0, 0] == -2.
    assert combine_values(edges, 'dispersion_penalty')[0, 0] < 0.
    assert np.isneginf(combine_values(edges, 'agreement')[0, 0])
    unanimous = np.array([[[2., -1.]], [[3., -2.]], [[4., -3.]]])
    assert combine_values(unanimous, 'agreement')[0, 0] == 3.
    np.testing.assert_array_equal(combine_values(unanimous, 'mean'), [[3., -2.]])
