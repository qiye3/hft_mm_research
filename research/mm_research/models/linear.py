"""合并中心化充分统计量，训练OLS、Ridge和Lasso，无需保存全训练矩阵。"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearStatistics:
    """均值及中心化交叉乘积，所有计算使用Float64。"""

    samples: int
    mean_x: np.ndarray
    mean_y: float
    xx: np.ndarray
    xy: np.ndarray
    yy: float


# 【核心逻辑124】将一个样本块转换为数值稳定的中心化充分统计量。
def linear_statistics(matrix: np.ndarray, target: np.ndarray) -> LinearStatistics:
    """输入共同有效的X/y，输出中心化统计，后续精确复用相同训练样本。"""
    mean_x = matrix.mean(axis=0)
    mean_y = float(target.mean())
    centered_x = matrix - mean_x
    centered_y = target - mean_y
    return LinearStatistics(len(target), mean_x, mean_y, centered_x.T @ centered_x,
                            centered_x.T @ centered_y, float(centered_y @ centered_y))


# 【核心逻辑125】使用块均值修正项合并逐日统计，避免大数相减造成精度损失。
def merge_statistics(left: LinearStatistics, right: LinearStatistics) -> LinearStatistics:
    """返回与直接拼接两块后中心化等价的统计量，内存与日期数量无关。"""
    count = left.samples + right.samples
    difference_x = right.mean_x - left.mean_x
    difference_y = right.mean_y - left.mean_y
    weight = left.samples * right.samples / count
    return LinearStatistics(
        count, left.mean_x + difference_x * right.samples / count,
        left.mean_y + difference_y * right.samples / count,
        left.xx + right.xx + weight * np.outer(difference_x, difference_x),
        left.xy + right.xy + weight * difference_x * difference_y,
        left.yy + right.yy + weight * difference_y ** 2,
    )


# 【非核心】从共同样本统计中提取一个模型组的列，保持组间评估样本可比。
def subset_statistics(statistics: LinearStatistics, indices: list[int]) -> LinearStatistics:
    """返回指定列的充分统计量，y及样本范围保持一致。"""
    return LinearStatistics(statistics.samples, statistics.mean_x[indices], statistics.mean_y,
                            statistics.xx[np.ix_(indices, indices)], statistics.xy[indices], statistics.yy)


# 【核心逻辑126】训练期标准化后拟合三种线性模型，Lasso使用Gram加速近端梯度。
def fit_linear_model(statistics: LinearStatistics, family: str, alpha: float) -> dict:
    """Ridge目标为均方误差加L2，Lasso目标为半均方误差加L1，截距不惩罚。"""
    scales = np.sqrt(np.maximum(np.diag(statistics.xx) / statistics.samples, 0))
    scales[scales == 0] = 1
    covariance = statistics.xx / statistics.samples / np.outer(scales, scales)
    covariance_xy = statistics.xy / statistics.samples / scales
    if family == "ols":
        coefficients = np.linalg.lstsq(covariance, covariance_xy, rcond=1e-10)[0]
    elif family == "ridge":
        coefficients = np.linalg.solve(covariance + alpha * np.eye(len(scales)), covariance_xy)
    elif family == "lasso":
        coefficients = np.zeros(len(scales))
        accelerated = coefficients.copy()
        momentum = 1.0
        lipschitz = max(float(np.linalg.eigvalsh(covariance)[-1]), 1.0)
        for iteration in range(100_000):
            step = accelerated - (covariance @ accelerated - covariance_xy) / lipschitz
            updated = np.sign(step) * np.maximum(np.abs(step) - alpha / lipschitz, 0)
            gradient = covariance @ updated - covariance_xy
            residual = np.where(updated != 0, np.abs(gradient + alpha * np.sign(updated)),
                                np.maximum(np.abs(gradient) - alpha, 0))
            if float(residual.max()) <= 1e-7:
                coefficients = updated
                break
            next_momentum = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2
            accelerated = updated + (momentum - 1) / next_momentum * (updated - coefficients)
            coefficients = updated
            momentum = next_momentum
        else:
            raise ValueError("Lasso加速近端梯度未达到KKT收敛条件")
    else:
        raise ValueError(f"未知线性模型 {family}")
    raw_coefficients = coefficients / scales
    return {
        "family": family, "alpha": alpha, "target_unit": "bps",
        "mean_x": statistics.mean_x.tolist(), "scale_x": scales.tolist(),
        "standardized_coefficients": coefficients.tolist(), "coefficients": raw_coefficients.tolist(),
        "intercept": float(statistics.mean_y - statistics.mean_x @ raw_coefficients),
        "training_samples": statistics.samples,
    }


# 【核心逻辑127】用独立区间统计评价预测误差、相关性与零预测基线。
def score_linear_model(statistics: LinearStatistics, model: dict) -> dict:
    """指标与逐行预测等价，常数预测的相关系数明确为空。"""
    coefficients = np.asarray(model["coefficients"])
    variance_sum = float(coefficients @ statistics.xx @ coefficients)
    covariance_sum = float(coefficients @ statistics.xy)
    mean_prediction = float(statistics.mean_x @ coefficients + model["intercept"])
    error_mean = mean_prediction - statistics.mean_y
    mse = max(0.0, (statistics.yy + variance_sum - 2 * covariance_sum) / statistics.samples + error_mean ** 2)
    denominator = np.sqrt(max(0.0, variance_sum * statistics.yy))
    return {
        "rows": statistics.samples, "mse_bps2": mse, "rmse_bps": float(np.sqrt(mse)),
        "ic": covariance_sum / denominator if denominator > 0 else None,
        "r2": 1 - mse / (statistics.yy / statistics.samples) if statistics.yy > 0 else None,
        "mean_prediction_bps": mean_prediction, "mean_target_bps": statistics.mean_y,
        "zero_prediction_mse_bps2": statistics.yy / statistics.samples + statistics.mean_y ** 2,
    }
