"""诊断指标区分全部预测与有效标签，不将无定义相关填零。"""

import numpy as np
from mm_research.evaluation.model_failures import finite_correlation, output_profile


# 【非核心】预测正值占比不按未来标签过滤，获利率只在有效未来标签上定义。
def test_profile_label_population():
    """第三行标签缺失不能从模型输出分布中消失。"""
    edge = np.array([-1., 2., 3., 4.])
    result = output_profile(edge, np.array([.2, .6, .7, .8]), np.array([-2., 1., np.nan, -3.]))
    assert result['positive_edge_fraction'] == .75
    assert result['positive_edge_win_rate'] == .5
    assert result['grid_rows'] == 4
    assert result['label_rows'] == 3
    assert finite_correlation(np.ones(4), edge) is None


# 【非核心】入场秒价格变化不能影响事前状态，质量边界显式保留缺失。
def test_entry_regime_uses_past_only():
    """用可手算价格验证时间偏移，再修改未来部分检查结果不变。"""
    from mm_research.evaluation.model_failures import entry_past_return

    market = np.ones((10, 10))
    market[:, 0] = np.arange(100., 110.)
    rows = np.array([5, 7])
    expected = np.log(np.array([104. / 102., 106. / 104.])) * 10000
    np.testing.assert_allclose(entry_past_return(market, rows, 2), expected)
    market[7:, 0] = 999.
    np.testing.assert_allclose(entry_past_return(market, rows, 2), expected)
    market[6, 9] = 2.
    assert np.isnan(entry_past_return(market, rows, 2)[1])


# 【非核心】按逐秒标记计数缺口，入场所在秒不得混入诊断窗口。
def test_entry_window_quality():
    """已知两个无效秒和一次恢复，随后修改未来行不影响统计。"""
    from mm_research.evaluation.model_failures import entry_window_quality

    market = np.ones((10, 10))
    market[:, 5] = np.arange(10) * 1_000_000_000
    market[:, 7] = 20_000_000
    market[2:4, 8] = 0.
    market[4:, 9] = 2.
    result = entry_window_quality(market, 7, 5)
    assert result['past_invalid_seconds'] == 2
    assert result['past_longest_invalid_seconds'] == 2
    assert result['past_segment_changes'] == 1
    assert result['seconds_since_segment_change'] == 3
    assert result['past_max_exchange_step_s'] == 1.
    market[7:, 8] = 0.
    assert entry_window_quality(market, 7, 5) == result


# 【非核心】回归评价必须排除无效标签，同时保留完整预测网格计数。
def test_regression_diagnostic_population(tmp_path):
    """线性可手算标签检查RMSE与分桶总数，最后一行无效极值不能污染结果。"""
    import polars as pl
    import yaml
    from mm_research.evaluation.model_failures import evaluate_regression_outputs

    prediction = np.linspace(-1., 1., 701)
    actual = prediction * 2
    actual[-1] = 10000.
    prediction_file = tmp_path / 'predictions.parquet'
    pl.DataFrame({
        'prediction_log_bps': prediction,
        'target_log_bps': actual,
        'evaluation_valid': np.arange(701) < 700,
    }).write_parquet(prediction_file)
    output = tmp_path / 'output'
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'output': str(output), 'predictions': {'example': str(prediction_file)}}))
    evaluate_regression_outputs(config)
    profile = pl.read_csv(output / 'regression_profiles.csv').row(0, named=True)
    assert profile['grid_rows'] == 701
    assert profile['label_rows'] == 700
    np.testing.assert_allclose(profile['rmse_bps'], np.sqrt(np.mean(prediction[:-1] ** 2)))
    assert pl.read_csv(output / 'regression_deciles.csv')['rows'].sum() == 700
