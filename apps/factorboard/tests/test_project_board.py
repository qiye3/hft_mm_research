"""只验证看板新增的分组、时间查询和统计汇总，不触发主项目任务。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from project_board.statistics import QUANTILES, aggregate_groups, confidence_interval, group_masks, group_statistics


class ProjectBoardTests(unittest.TestCase):
    # 【非核心】十二组是十个互斥组加重叠两尾，组计数不能一起相加。
    def test_ten_deciles_and_overlapping_tails(self):
        """1000个严格递增数形成十组各100点，两尾各10点。"""
        values = np.arange(1000, dtype=float)
        thresholds = np.quantile(values, QUANTILES, method='linear').tolist()
        masks = group_masks(values, thresholds)
        self.assertEqual([int(mask.sum()) for mask in masks], [10] + [100] * 10 + [10])
        self.assertTrue(np.all(masks[0] <= masks[1]))
        self.assertTrue(np.all(masks[-1] <= masks[-2]))
        np.testing.assert_array_equal(np.sum(masks[1:-1], axis=0), np.ones(1000))

    # 【非核心】并列值不硬拆；固定名称的空组返回null。
    def test_ties_keep_empty_groups(self):
        """常数因子只占G1与重叠左尾，组内IC不可计算。"""
        frame = pl.DataFrame({'value': [0.] * 100, 'target': list(range(100)), 'label_valid': [True] * 100})
        rows = group_statistics(frame, [0.] * 11)
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[1]['count'], 100)
        self.assertIsNone(rows[1]['ic'])
        self.assertIsNone(rows[2]['bps'])

    # 【非核心】按日等权收益，与所有样本合并均值不同。
    def test_equal_day_aggregation(self):
        """两个样本数不同的日，收益均值为2而不是样本加权结果。"""
        days = []
        for count, target in [(10, 1.), (90, 3.)]:
            frame = pl.DataFrame({'value': [0.] * count, 'target': [target] * count, 'label_valid': [True] * count})
            days.append({'groups': {'daily': {'1000': group_statistics(frame, [0.] * 11)}}})
        aggregate = aggregate_groups(days, 'daily', 1000)
        self.assertEqual(aggregate[1]['bps'], 2.)
        self.assertEqual(aggregate[1]['average_count'], 50.)
        self.assertEqual(aggregate[1]['count'], 100)

    # 【非核心】固定种子的区间可复现，常数序列保留真实零宽度。
    def test_block_interval(self):
        """重采样不引入虚构噪声，输入不足时返回null。"""
        np.testing.assert_allclose(confidence_interval([.1] * 30), [.1, .1])
        self.assertIsNone(confidence_interval([.1, None]))
        self.assertEqual(confidence_interval(list(range(30))), confidence_interval(list(range(30))))


if __name__ == '__main__':
    unittest.main()
