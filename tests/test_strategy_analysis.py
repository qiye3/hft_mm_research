"""核对资金费方向、价格区间和Markout的时间边界。"""

import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from mm_research.strategies.analysis import funding_cashflows, add_fill_markouts


class StrategyAnalysisTests(unittest.TestCase):
    # 【非核心】长仓付费、短仓收款，结算时只计已经发生的交易所成交。
    def test_signed_funding_and_price_bounds(self):
        """三个结算点依次为多仓、空仓、空仓位，分钟价格区间按现金流符号处理。"""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            records = [{'fundingTime': timestamp, 'fundingRate': .001, 'markPrice': 100,
                        'markPriceLow': 90, 'markPriceHigh': 110} for timestamp in [1000, 2000, 3000]]
            (output / 'funding_2026-10-21.json').write_text(json.dumps({'records': records}))
            fills = pl.DataFrame({'exchange_timestamp_ns': [500_000_000, 1_500_000_000, 2_500_000_000],
                                  'side': [1, -1, 1], 'quantity': [1., 2., 1.]})
            cash, status = funding_cashflows(output, fills, '2026-10-21')
            self.assertEqual(status, 'official_archive_with_mark_interval')
            self.assertEqual(cash['position'].to_list(), [1., -1., 0.])
            self.assertAlmostEqual(cash['funding_payment'].sum(), 0)
            self.assertAlmostEqual(cash['funding_payment_low'].sum(), -.02)
            self.assertAlmostEqual(cash['funding_payment_high'].sum(), .02)

    # 【非核心】资金费文件缺失时保留未覆盖状态，不把未知收费写成零。
    def test_unavailable_funding(self):
        """缺少输入时表为空且状态明确，交易费用核算可独立保存。"""
        with tempfile.TemporaryDirectory() as directory:
            cash, status = funding_cashflows(Path(directory), pl.DataFrame(), '2026-10-21')
            self.assertEqual(status, 'unavailable')
            self.assertTrue(cash.is_empty())

    # 【非核心】Markout取到期前最近可见盘口，不读取更晚报价，数据尾端保持空值。
    def test_markout_backward_alignment(self):
        """买卖方向使用相反符号，超过数据覆盖范围的五秒结果为空。"""
        fills = pl.DataFrame({'exchange_timestamp_ns': [0, 0], 'side': [1, -1], 'price': [100., 100.]})
        book = pl.DataFrame({'local_timestamp_ns': [1_000_000_000, 1_020_000_000, 1_100_000_000],
                             'mid': [100., 101., 999.]})
        result = add_fill_markouts(fills, book)
        self.assertAlmostEqual(result['markout_1s_bps'][0], 100.)
        self.assertAlmostEqual(result['markout_1s_bps'][1], -100.)
        self.assertEqual(result['markout_5s_bps'].null_count(), 2)


if __name__ == '__main__':
    unittest.main()
