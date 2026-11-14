"""概率评价与成块配对统计的独立手算参考。"""

import unittest

import numpy as np
import polars as pl

from mm_research.evaluation.fee_models import binary_scores, weekly_error_intervals


class FeeModelEvaluationTest(unittest.TestCase):
    # 【非核心】并列排序不能人为增加AUC，概率误差应与独立二元公式一致。
    def test_known_binary_scores(self) -> None:
        """已知两正两负、一个并列对，AUC为0.875、Brier为0.13。"""
        actual = np.array([0., 0., 1., 1.])
        probability = np.array([.1, .5, .5, .9])
        result = binary_scores(actual, probability)
        self.assertAlmostEqual(result["auc"], .875)
        self.assertAlmostEqual(result["brier"], .13)
        self.assertAlmostEqual(result["logloss"], -(np.log(.9) + np.log(.5)) / 2)
        self.assertIsNone(binary_scores(np.ones(4), probability)["auc"])

    # 【非核心】各周误差增量相同时，任意成块重采样都应给出同一个精确差值。
    def test_constant_paired_week_difference(self) -> None:
        """样本数不等也不能改变常数增量，确保使用相同周内样本加权。"""
        daily = pl.DataFrame({
            "group": ["e1"] * 4, "side": [1] * 4,
            "date": ["2026-03-23", "2026-03-24", "2026-03-30", "2026-04-06"],
            "count": [10, 20, 40, 100], "delta_logloss": [-.02] * 4, "delta_brier": [.01] * 4,
        })
        results = weekly_error_intervals(daily, "e0", 100, 17)
        for result, expected in zip(results, [-.02, .01]):
            self.assertEqual(result["weeks"], 3)
            for field in ["delta", "lower_95", "upper_95"]:
                self.assertAlmostEqual(result[field], expected)


if __name__ == "__main__":
    unittest.main()
