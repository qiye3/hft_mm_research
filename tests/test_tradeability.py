"""180秒研究的因果窗口、非重叠机会和完整交易核算回归。"""

import numpy as np

from mm_research.multihorizon.common import HORIZONS, SECOND
from mm_research.multihorizon.inventory import simulate_inventory
from mm_research.strategies.tradeability import independent_opportunities, market_states
from mm_research.strategies.tradeability_execution import active_opportunity_reference, snapshot_round_trips


# 【非核心】机会跨午夜继续等待，连续强信号不能每秒重复视为独立交易。
def test_opportunities_keep_holding_time_across_midnight():
    """午夜前30秒触发后，下一次机会在午夜后150秒出现。"""
    timestamps = (np.arange(400) + 86400 - 30) * SECOND
    selected = np.ones(400, dtype=bool)
    opportunities = independent_opportunities(timestamps, selected, 180 * SECOND)
    np.testing.assert_array_equal(np.flatnonzero(opportunities), [0, 180, 360])


# 【非核心】未来盘口变动不影响此前波动率，子区间保留此前60秒预热。
def test_market_states_use_only_past_and_keep_prefix():
    """整体计算与跨日分块读取在相同时间位置得到相同波动。"""
    manifest = {'market_columns': ['mid_price', 'bid_price_1', 'ask_price_1', 'book_valid']}
    mid = 100000 + np.sin(np.arange(160) / 4)
    market = np.column_stack([mid, mid - .1, mid + .1, np.ones(160)])
    config = {'state_lookback_s': 60, 'state_min_samples': 48}
    complete = market_states(market, manifest, 0, 160, config)
    block = market_states(market, manifest, 80, 140, config)
    np.testing.assert_allclose(
        complete['volatility_60s_bps'].to_numpy()[80:140],
        block['volatility_60s_bps'].to_numpy(), atol=1e-10,
    )
    changed = market.copy()
    changed[100:, :3] *= 2
    changed_states = market_states(changed, manifest, 0, 160, config)
    np.testing.assert_allclose(
        complete['volatility_60s_bps'].to_numpy()[:100],
        changed_states['volatility_60s_bps'].to_numpy()[:100], equal_nan=True,
    )


# 【非核心】零方向保持空仓，一次强信号触发后实际持有满180秒并结算双边费用。
def test_snapshot_gate_does_not_open_on_zero_direction():
    """固定持有期间信号归零仍应按持有期退出，验证实际共用模拟器语义。"""
    count = 500
    books = np.ones((count, 40))
    books[:, :10] = 100000 - np.arange(10) * .1
    books[:, 20:30] = 100000.1 + np.arange(10) * .1
    predictions = np.zeros((count, len(HORIZONS)))
    index = int(np.flatnonzero(HORIZONS == 180)[0])
    predictions[120, index] = 1
    funding = (np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float))
    ledger, fills, position = simulate_inventory(
        predictions, books, np.arange(count, dtype=float), np.ones(len(HORIZONS)),
        0, index, 180, 1., 0., 1., .002, .0005, 100., .001, 180, 10000., *funding,
    )
    np.testing.assert_array_equal(fills[:, 0], [120., 300.])
    assert position == 0
    expected = .002 * (100000 - 100000.1) - .0005 * .002 * (100000 + 100000.1)
    np.testing.assert_allclose(ledger[-1, 7], expected, atol=1e-10)


# 【非核心】部分平仓仍属于同一笔往返，资金费根据结算前实际剩余仓位计算。
def test_round_trip_partial_close_and_funding():
    """两次减仓分别收取手续费，资金费发生在第一次减仓之后。"""
    fills = np.array([
        [10., 1., .002, 100000., .1, .002, 0.],
        [190., -1., .001, 100100., .05005, .001, 0.],
        [191., -1., .001, 100100., .05005, 0., 0.],
    ])
    funding = (np.array([190.5]), np.array([.0001]), np.array([100000.]))
    trades = snapshot_round_trips(fills, funding)
    assert trades.height == 1
    np.testing.assert_allclose(trades['funding_paid_usdt'][0], .01)
    np.testing.assert_allclose(trades['net_pnl_usdt'][0], .2 - .2001 - .01, atol=1e-10)


# 【非核心】主动成交只能使用到达之前的盘口，之后的异常价格不能影响本次成交。
def test_active_reference_never_uses_future_snapshot(tmp_path, monkeypatch):
    """开仓和平仓都在两份快照之间到达，核对逐档价格与双边手续费。"""
    import json
    import polars as pl
    from mm_research.strategies import tradeability_execution

    source = tmp_path / 'source'
    (source / 'e0').mkdir(parents=True)
    (source / 'e0/manifest.json').write_text(json.dumps({'dates': ['2026-03-23']}))
    (source / 'e0/splits.json').write_text(json.dumps({'segments': {}}))
    standard = tmp_path / 'standard'
    (standard / 'snapshots').mkdir(parents=True)
    columns = {'exchange_timestamp_ns': [SECOND, SECOND + 2_000_000, 181 * SECOND, 181 * SECOND + 2_000_000]}
    for side in ['bid', 'ask']:
        for level in range(1, 11):
            offset = level * .1 if side == 'ask' else -(level - 1) * .1
            columns[f'{side}_price_{level}'] = [100000 + offset, 200000 + offset, 100200 + offset, 300000 + offset]
            columns[f'{side}_quantity_{level}'] = [1.] * 4
    pl.DataFrame(columns).write_parquet(standard / 'snapshots/2026-03-23.parquet')

    # 【非核心】该合成机会不跨资金费时刻，避免外部下载影响盘口时序测试。
    def no_funding(*arguments):
        """返回明确的空资金费日。"""
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    monkeypatch.setattr(tradeability_execution, 'funding_inputs', no_funding)
    config = {
        'standard_folder': str(standard), 'hold_s': 180,
        'execution': {'entry_latency_ns': 1_000_000, 'quantity_step_btc': .001, 'taker_fee': .0005},
    }
    opportunities = pl.DataFrame({
        'candidate_id': ['one'], 'decision_ts_ns': [SECOND], 'side': [1], 'submitted_qty_btc': [.002],
    })
    result = active_opportunity_reference(config, source, opportunities, '2026-03-23')
    expected = ((100200 - 100000.1) - (100200 + 100000.1) * .0005) / 100000.1 * 10000
    assert result['active_status'][0] == 'complete'
    np.testing.assert_allclose(result['active_net_return_bps'][0], expected, atol=1e-10)
