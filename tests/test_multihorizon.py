"""多期限研究的标签、分区、预处理与滚动仓位边界。"""

import datetime as dt
import unittest

import numpy as np

from mm_research.multihorizon.common import model_inputs
from mm_research.multihorizon.data import cumulative_labels, make_splits


class MultihorizonTests(unittest.TestCase):
    # 【非核心】已知指数价格的对数bps逐期限精确恢复，增量累加误差小于1e-9。
    def test_log_labels_and_reconstruction(self):
        """每秒上涨1 log bps，未来期限收益应严格等于期限秒数。"""
        mid = np.exp(np.arange(1000) / 10000)
        targets = cumulative_labels(mid, np.ones(1000, dtype=bool), np.zeros(1000))
        expected = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300])
        np.testing.assert_allclose(targets[0], expected, atol=1e-9, rtol=0)
        increments = np.diff(np.r_[0, targets[0]])
        np.testing.assert_allclose(increments.cumsum(), targets[0], atol=1e-9, rtol=0)

    # 【非核心】内部缺口和连续区段改变都会阻断标签，序列末尾保持缺失。
    def test_gaps_and_tail(self):
        """有效首尾也不能跨过无效盘口或来源切换计算收益。"""
        mid = np.arange(1000) + 100.
        valid = np.ones(1000, dtype=bool)
        valid[5] = False
        segment = np.zeros(1000)
        segment[100:] = 1
        labels = cumulative_labels(mid, valid, segment)
        self.assertTrue(np.isnan(labels[0, 3]))
        self.assertTrue(np.isnan(labels[99, 0]))
        self.assertTrue(np.isnan(labels[-1]).all())

    # 【非核心】所有OOF训练标签结束后才进入预测折，整日分区保持先后顺序。
    def test_forward_split_boundaries(self):
        """531日固定前318日训练，训练前缀与预测区之间隔离300秒。"""
        dates = [(dt.date(2025, 5, 9) + dt.timedelta(days=day)).isoformat() for day in range(531)]
        splits = make_splits(dates)
        self.assertEqual(splits['segments']['train']['stop_row'], 318 * 86400)
        for fold in splits['oof_folds']:
            self.assertEqual(fold['train_stop'] + 300, fold['predict_start'])

    # 【非核心】变更未来行不影响过去输入，缺失填补严格使用传入的训练统计量。
    def test_past_inputs_and_missing_channels(self):
        """输出顺序为标准化数值后接缺失标记，未来扰动不改变当前批次。"""
        matrix = np.array([[1., np.nan], [999., 999.]], dtype='float32')
        mean = np.array([1., 2.], dtype='float32')
        scale = np.ones(2, dtype='float32')
        before = model_inputs(matrix, np.array([0]), mean, scale)
        matrix[1] = -999.
        np.testing.assert_array_equal(before, model_inputs(matrix, np.array([0]), mean, scale))
        np.testing.assert_array_equal(before, [[0., 0., 0., 1.]])


if __name__ == '__main__':
    unittest.main()
