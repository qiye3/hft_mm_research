"""仅用训练日统计的Gen041筛选规则手算测试。"""

import unittest

import polars as pl

from mm_research.models.gen041_selection import select_from_daily


class Gen041SelectionTest(unittest.TestCase):
    # 【非核心】同一经济家族不能占满名额，方向摇摆或覆盖不足不进入候选。
    def test_sign_stability_and_family_limit(self) -> None:
        """固定四日相关系数，稳定正负方向均可入选，验证并列之外的明确排序。"""
        values = {
            "book_obi_l1_mean_300s": [.2] * 4,
            "book_obi_l1_mean_900s": [.15] * 4,
            "book_obi_l1_mean_1800s": [.12] * 4,
            "trade_aggr_qtyimb_300s": [.3, -.3, .3, -.3],
            "trade_aggr_qtyimb_900s": [.4, .4, .4, None],
            "trade_aggr_qtyimb_1800s": [-.1] * 4,
        }
        columns = list(values)
        frame = pl.DataFrame({
            "samples": [100] * 4,
            **values,
            **{f"{column}_count": [100] * 4 for column in columns},
        })
        config = {
            "minimum_daily_samples": 100,
            "minimum_coverage": .8,
            "minimum_day_fraction": .8,
            "minimum_sign_agreement": .6,
            "maximum_factors": 12,
            "maximum_per_family": 2,
        }
        _, selected = select_from_daily(frame, columns, config)
        self.assertEqual(selected, ["book_obi_l1_mean_300s", "book_obi_l1_mean_900s", "trade_aggr_qtyimb_1800s"])


if __name__ == "__main__":
    unittest.main()
