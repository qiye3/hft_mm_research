"""交易路径诊断的方向、缺口和区间边界。"""

import numpy as np

from mm_research.evaluation.trade_paths import price_path


# 【非核心】合成先涨后跌再涨路径，验证多空极值和无效路径不被拼接。
def test_price_path() -> None:
    market = np.zeros((4, 10))
    market[:, 0] = [100, 101, 99, 102]
    market[:, 8] = 1
    long_path = price_path(market, 0, 3, 1)
    assert long_path['mfe_time_s'] == 3
    assert long_path['mae_time_s'] == 2
    assert np.isclose(long_path['mid_return_bps'], np.log(1.02) * 10000)
    assert price_path(market, 0, 3, -1)['mfe_time_s'] == 2
    assert not price_path(market, 0, 4, 1)['path_valid']
    market[1, 9] = 1
    assert not price_path(market, 0, 3, 1)['path_valid']
    market[:, 9] = 0
    market[1, 8] = 0
    assert not price_path(market, 0, 3, 1)['path_valid']
