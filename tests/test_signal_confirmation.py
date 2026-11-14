"""连续信号只控制新入场，确认时钟不能使用未来行。"""
import numpy as np
from mm_research.strategies.account_sizing import confirmed_entry


# 【非核心】覆盖方向变化、零信号、前缀不足及追加未来数据不改变历史结果。
def test_confirmation_is_causal():
    """第三个连续同向秒才准入；方向变化重新计数。"""
    signal = np.array([1., 1., 1., 0., -1., -1., -1., 1.])
    expected = [False, False, True, False, False, False, True, False]
    np.testing.assert_array_equal(confirmed_entry(signal, 3), expected)
    np.testing.assert_array_equal(confirmed_entry(signal[:5], 3), expected[:5])
    np.testing.assert_array_equal(confirmed_entry(signal, 1), signal != 0.)
