"""固定本金年化与敞口统计不缩短时间、不缩小分母。"""

import numpy as np

from mm_research.strategies.account_sizing import account_metrics


# 【非核心】一年账户盈利1000USDT、初始本金10000USDT，对应10%年化。
def test_account_return_keeps_calendar_and_capital():
    """即使中途无仓位，整个365天仍计入年化；敞口使用实际净权益。"""
    ledger = np.zeros((3, 11))
    ledger[:, 0] = [0., 180 * 86400., 365 * 86400. - 1.]
    ledger[:, 7] = [0., -500., 1000.]
    ledger[:, 3] = [0., .05, 0.]
    books = np.full((3, 40), 100000.)
    result = account_metrics(ledger, books, 10000.)
    np.testing.assert_allclose(result['annualized_return'], .1)
    np.testing.assert_allclose(result['max_drawdown_fraction'], .05)
    np.testing.assert_allclose(result['peak_gross_leverage'], 5000. / 9500.)
    assert result['duration_days'] == 365.
