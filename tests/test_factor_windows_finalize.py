"""四档产物验收和训练期前向窗口选择的直接回归。"""

import datetime as dt
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from mm_research.data.storage import write_parquet

specification = importlib.util.spec_from_file_location(
    'factor_windows_finalize', Path(__file__).resolve().parents[1] / 'scripts/finalize_factor_windows.py')
finalizer = importlib.util.module_from_spec(specification)
specification.loader.exec_module(finalizer)


class FactorWindowFinalizationTests(unittest.TestCase):
    # 【非核心】验证期更高的相关不能改变训练前向三折选出的窗口。
    def test_selection_uses_forward_training_only(self):
        """600秒窗口三折Rank IC更高，验证期偏向180秒的统计被排除。"""
        dates = [(dt.date(2025, 5, 9) + dt.timedelta(days=index)).isoformat() for index in range(20)]
        daily = []
        summaries = []
        for window, rank_ic in [(180, .03), (600, .05)]:
            factor = f'book_obi_l1_mean_{window}s'
            daily.extend({'factor': factor, 'date': date, 'rank_ic': rank_ic} for date in dates)
            for split, value in [('train', rank_ic), ('val_select', 1. if window == 180 else -1.)]:
                summaries.append({'factor': factor, 'split': split, 'sampling': 'common',
                                  'horizon_ns': 180_000_000_000, 'rank_ic_mean': value, 'valid_ic_days': 20})
        selected, folds = finalizer.select_windows(pl.DataFrame(summaries), pl.DataFrame(daily), dates)
        self.assertEqual(selected['factor'].to_list(), ['book_obi_l1_mean_600s'])
        self.assertEqual(folds.height, 6)
        self.assertAlmostEqual(selected['forward_rank_ic'][0], .05)

    # 【非核心】全日Parquet页脚必须具有完整行数、UTC端点与Float64因子列。
    def test_complete_day_footer(self):
        """使用真实写入入口核对验收读出的完整网格。"""
        date = '2026-07-12'
        first = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
        name = 'book_obi_l1_mean_600s'
        table = pl.DataFrame({'symbol': ['BTCUSDT'] * 86400, 'date': [date] * 86400,
                              'timestamp_ns': first + np.arange(86400, dtype=np.int64) * 1_000_000_000,
                              name: np.zeros(86400)})
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_parquet(table.lazy(), folder / f'{name}.parquet',
                          {'kind': 'factor', 'dataset': {'dates': [date]}, 'definition': {'dates': [date]}})
            result = finalizer.inspect_factor((folder, 'gen001', name, [date]))
            self.assertEqual(result['rows'], 86400)


if __name__ == '__main__':
    unittest.main()
