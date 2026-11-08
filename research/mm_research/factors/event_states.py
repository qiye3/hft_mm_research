"""有界历史状态计算，仅供需要逐位置状态转换的突破因子调用。"""

import numpy as np
from numba import njit


# 【核心逻辑96】只用已发生的价格更新维护两秒确认状态，触发桶不提前记失败。
@njit(cache=True)
def breakout_states(mid: np.ndarray, allowed: np.ndarray, tick_size: float) -> np.ndarray:
    """返回九个100ms状态，历史高低点排除当前桶且至少40个有效位置。"""
    output = np.full((len(mid), 9), np.nan)
    up_start = -1
    down_start = -1
    up_boundary = 0.0
    down_boundary = 0.0
    previous_up = False
    previous_down = False
    for index in range(len(mid)):
        if not allowed[index] or not np.isfinite(mid[index]):
            up_start = -1
            down_start = -1
            previous_up = False
            previous_down = False
        if not np.isfinite(mid[index]):
            continue
        high = -np.inf
        low = np.inf
        valid = 0
        for previous in range(max(0, index - 50), index):
            if np.isfinite(mid[previous]):
                high = max(high, mid[previous])
                low = min(low, mid[previous])
                valid += 1
        if valid < 40:
            continue
        up = mid[index] > high
        down = mid[index] < low
        up_failed = 0.0
        down_failed = 0.0
        if up_start >= 0:
            if index - up_start > 20:
                up_start = -1
            elif mid[index] <= up_boundary:
                up_failed = 1.0
                up_start = -1
        if down_start >= 0:
            if index - down_start > 20:
                down_start = -1
            elif mid[index] >= down_boundary:
                down_failed = 1.0
                down_start = -1
        if up and not previous_up and up_start < 0:
            up_start = index
            up_boundary = high
        if down and not previous_down and down_start < 0:
            down_start = index
            down_boundary = low
        up_distance = max(mid[index] - high, 0) / tick_size
        down_distance = max(low - mid[index], 0) / tick_size
        up_value = 1.0 if up else 0.0
        down_value = 1.0 if down else 0.0
        output[index, 0] = up_value
        output[index, 1] = down_value
        output[index, 2] = up_value - down_value
        output[index, 3] = up_distance
        output[index, 4] = down_distance
        output[index, 5] = up_distance - down_distance
        output[index, 6] = up_failed
        output[index, 7] = down_failed
        output[index, 8] = up_failed - down_failed
        previous_up = up
        previous_down = down
    return output
