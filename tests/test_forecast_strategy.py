"""冻结预测组合与新策略粗网格的直接回归。"""

import numpy as np
import polars as pl

from mm_research.strategies.forecast_research import ensemble_prediction
from mm_research.strategies.forecast_execution import strategy_variants, variant_name


# 【非核心】预测固定等权平均，未来标签和评价有效性不改变信号输出行。
def test_fixed_ensemble_keeps_full_grid(tmp_path):
    """尾行没有标签仍输出预测，组合不按模型事后成绩选择权重。"""
    files = []
    for index, values in enumerate([[1., 3., 5.], [3., 5., 7.]]):
        file = tmp_path / f'{index}.parquet'
        pl.DataFrame({'timestamp_ns': [0, 1, 2], 'prediction_log_bps': values,
                      'target_log_bps': [1., 2., float('nan')]}).write_parquet(file)
        files.append(file)
    result = ensemble_prediction(files)
    np.testing.assert_array_equal(result['prediction_log_bps'].to_numpy(), [2., 4., 6.])


# 【非核心】三日先导案例可以直接复用于完整开发网格，每个实验名必须唯一。
def test_pilot_is_subset_of_development_grid():
    """等待、宽度、持仓和双周期开关进入唯一实验名。"""
    config = {'signals': [{'id': 'long', 'horizon_s': 180}, {'id': 'short', 'horizon_s': 30}]}
    pilot = {variant_name(case) for case in strategy_variants(config, pilot=True)}
    development = [variant_name(case) for case in strategy_variants(config, pilot=False)]
    assert pilot <= set(development)
    assert len(development) == len(set(development))


# 【非核心】并发启动只初始化一次冻结配置，后续读取不触发共享盘重复替换。
def test_concurrent_context_writes_config_once(tmp_path, monkeypatch):
    """文件锁覆盖检查与首次发布，全部工作进程读取同一份配置。"""
    from concurrent.futures import ThreadPoolExecutor
    import yaml
    from mm_research.strategies import forecast_research

    filename = tmp_path / 'research.yaml'
    config = {'source': 'frozen_source', 'output': str(tmp_path / 'output')}
    filename.write_text(yaml.safe_dump(config))
    original = forecast_research.write_json_atomic
    writes = []

    # 【非核心】记录真实原子发布次数，校验初始化不会因工作进程数增加而重复。
    def record_write(path, value):
        writes.append(path)
        original(path, value)

    monkeypatch.setattr(forecast_research, 'write_json_atomic', record_write)
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(forecast_research.research_context, [filename] * 8))
    assert len(writes) == 1
    assert all(result[0] == config for result in results)
