"""分块OOF校准与完整曲线风险统计的回归验证。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from mm_research.multihorizon.evaluation import calibrate_model


class MultihorizonAnalysisTests(unittest.TestCase):
    # 【非核心】分块充分统计与已知仿射关系一致，缺失标签行不进入阈值或风险估计。
    def test_streaming_oof_calibration(self):
        """三个OOF文件按序读取，预测为真实值两倍加常数，校准应恢复真实bps。"""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            data = output / 'e0'
            data.mkdir()
            (data / 'manifest.json').write_text('{}')
            (data / 'splits.json').write_text('{}')
            random = np.random.default_rng(42)
            truth = random.normal(size=(300, 16))
            forecast = truth * 2 + 1.2
            truth[7] = np.nan
            np.save(data / 'y.npy', truth)
            np.save(data / 'x.npy', np.zeros((300, 2), dtype='float32'))
            np.save(data / 'market.npy', np.zeros((300, 10)))
            folder = output / 'e1/predictions/M1'
            folder.mkdir(parents=True)
            for fold in range(3):
                rows = np.arange(fold * 100, (fold + 1) * 100)
                pl.DataFrame({'row': rows, 'pred_curve_bps': forecast[rows]}).write_parquet(folder / f'oof_{fold}.parquet')
            calibration = calibrate_model(output, 'M1')
            np.testing.assert_allclose(calibration['slope'], .5, atol=1e-12)
            np.testing.assert_allclose(calibration['intercept_bps'], -.6, atol=1e-12)
            expected = np.quantile(np.delete(truth[:, 0], 7), [.2, .4, .6, .8])
            np.testing.assert_allclose(calibration['quantile_thresholds'][0], expected, atol=1e-12)
            self.assertEqual(calibration['oof_rows'], 299)
            self.assertLess(np.max(np.abs(calibration['increment_error_covariance_bps2'])), 1e-20)


if __name__ == '__main__':
    unittest.main()
