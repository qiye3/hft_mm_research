# 2026-11-03 — M1 报价、委托与交易账本

运行路径于 2026-11-04 适配当前目录布局。

## 目标与当前链路

M1 使用 M0 的同一份 2025-02-01 BTCUSDT 数据，完成“本地可见盘口 → 报价 → 原生委托 → 原生成交 → 费用/库存 → PnL”。不开发 Alpha、机器学习、库存 skew 或自定义撮合。

```text
EventReaderCache → Backtest::elapse
    ├─ Local::process → 本地 BBO
    ├─ PartialFillExchange::process → 原生排队/成交
    └─ RecordingLocal::process_recv_order
          ├─ 原生 Local 更新订单和 StateValues
          └─ TradeRecorder 逐条记录回报并核对账本
定时决策 → MarketMaker → Backtest::submit_* / cancel
```

## 文件与职责

| 文件 | 关键接口与职责 |
| --- | --- |
| `src/mm/market_maker.hpp`、`src/mm/market_maker.cpp` | `fixed_quotes`、`MarketMaker::on_decision/quote_side/reservations`；只计算目标报价并调用原生 API，不另存订单状态机 |
| `src/mm/trade_recorder.hpp`、`src/mm/trade_recorder.cpp` | `TradeRecorder::response/reconcile/checkpoint/finish`；记录逐笔流水、平均成本法已实现/未实现收益，与原生 StateValues 对账 |
| `src/mm/detail/trading_engine.hpp` | 使用原生 `OrderBus`、`Local`、`PartialFillExchange`、`ConstantLatency` 和 `RiskAdverseQueueModel`；RecordingLocal 是公开 LocalProcessor 接口的转发适配，不重新实现订单生命周期 |
| `src/apps/baseline_mm.cpp` | CLI 入口，调用 `run_baseline` |
| `config/baseline_mm.yaml` | 报价半宽、每单数量、库存上限、手续费、延迟、更新模式、停止报价时间 |
| `tests/test_market_maker.cpp` | 原生引擎驱动的成交、费用、部分成交、撤单竞态与敞口回归测试 |
| `scripts/analyze_baseline.py` | 串行运行 A/B/C 实验、从 CSV 独立对账、计算本地接收时刻后 1 秒 markout |

## 报价与风控

默认每 100 ms 评估一次。买价为 `floor(mid_tick - half_width_ticks)`，卖价为 `ceil(mid_tick + half_width_ticks)`，向外取整以免无意缩窄报价。参考中间价来自 Local Processor，绝不读取 Exchange Processor 的未来盘口。使用 `GTX + Limit`，交易所接收时若已经穿价，交由原引擎使订单 Expired，不转成 Taker。

每侧最多一笔未终结订单。新单在途、已接受未成交和撤单在途，都保留剩余数量预算。策略风险约束为：

```text
native_position + sum(unconfirmed_or_active_buy_leaves) <= max_position
native_position - sum(unconfirmed_or_active_sell_leaves) >= -max_position
```

撤单请求不会释放预算，也不会立即发替代单；必须观察原生确认后的状态。部分成交仅减少真实剩余数量，新增库存计入相应方向的可用空间。订单 ID 递增仅用于唯一性，不充当自定义订单管理器。终结订单通过 `Backtest::clear_inactive_orders` 清理。

价格模式以“上次更新目标时的中间价”为锚，累计变动达到阈值才重定目标，检查间隔仍为 100 ms。被成交的一侧可以按现有目标补单，风险限制始终生效。定时模式也不撤掉价格没有变化的订单。

数据结束前一秒停止新增报价并撤单；处理完所有原生回报后验证没有活动或在途订单。尾仓保留，按最后本地有效 Mid 盯市，不制造期末平仓。短暂空簿期间不发新单；账本使用最近有效的本地 Mid。

## 精确回报与上游兼容处理

固定依赖仍是 `e0d2f7887c68755b220d9e69425d61d27e5b6432`，第三方源码没有修改。

上游 `include/hbt/backtest/local.hpp::Local::process_recv_order` 会先依据 `status == Filled/PartiallyFilled` 调用 `state_.apply_fill`，之后才检查 `req == Rejected`。`exchange.hpp::ack_cancel` 在订单已经成交时，可能把撤单请求中的旧 Partial/Filled 状态、旧 exec_qty 原样带回，并设置 `req=Rejected`。这不是一笔新的成交，却可能重复计账。

已通过两类合成用例确认：

1. 直接给固定版本的原生 Local 一条带旧 Partial 状态的拒绝回报，即使没有真实新成交也会增加库存。
2. 真正下 3 个 lot 买单，先成交 1 个 lot，再发撤单；剩余 2 个 lot 在撤单到达前成交。正确结果为累计 3 个 lot、两笔成交，随后仅收到撤单被拒。

`RecordingLocal` 在框架公开 OrderBus 队列边界，只对 **req=Rejected 且 status 为旧 Filled/Partial** 的回报，规范化其非经济计账字段：送给 Local 的 status=None、exec_qty=0；原始回报保留在 orders.csv。Local 本身的拒绝处理分支、订单总线、延迟、队列和成交模型均保留原实现。每次处理后立即将现金、费用、成交数、成交量、成交额和仓位与原生 StateValues 对账，不用错误账本继续实验。

本次默认每单恰为一个最小 lot，因此真实日没有部分成交，兼容分支未触发（各组 normalized_rejections=0）。多 lot 部分成交和上述竞态由独立合成回归覆盖。

## 账本口径

原生 Local/State 是实际策略持仓、现金流和费用的来源；TradeRecorder 的独立账本用于分解 PnL 和即时核对，不替代它。平均成本法处理加仓、减仓、平仓和翻向。

```text
net_pnl = native_balance + native_position * local_mid - native_fee
         = realized_gross + unrealized - fees
account_equity = initial_capital + net_pnl
inventory_value = position * local_mid
```

`balance` 是框架累计买卖交易现金流，不是 Binance 保证金余额。费用为逐笔成交名义金额乘 Maker/Taker 费率，负费率表示返佣。单位为 BTC 和 USDT，线性合约乘数为 1。C++ 在每批同时间戳原生回报后对账；即使两笔相反成交在一个决策周期内抵消，也会分别导出，不从净库存差推断成交。

- `orders.csv`：submit、cancel_request、每条原始 response，包括订单 ID、方向、报价、原始数量、剩余数量、状态、请求状态、执行价格/数量及双时间戳。请求记录的 exch_timestamp 为当时已知原订单时间，不是尚未发生的交易所确认时间。
- `fills.csv`：每次真实成交的数量、费用、成交后账本库存、现金流、平均成本、已实现收益和净 PnL。
- `equity.csv`：每个决策点及每次成交的本地 Mid、库存市值、费用、PnL、权益和运行回撤。成交点的 reserved_buy/sell 留空，避免把同批回报结束后的原生订单状态错配到中间成交；决策点记录真实原生剩余数量。
- `summary.json`：聚合指标、参数、回放事件数和对账状态。

## 实验与验证

实际结果见 [七组单日实验](m1_2025-02-01/report.md)。Python 从导出 CSV 检查唯一订单、累计成交量、原始回报计数、逐笔费用/平均成本、现金与仓位、逐点 PnL、风险预留和回撤。每组必须完整消费 M0 同一天的双侧行情；没有抽样或缩短区间。

原生引擎单元用例验证：向外 Tick 取整、GTX 失效、买卖往返净收益、负手续费返佣、连续部分成交、库存硬上限、撤单在途不补单、部分成交撤单竞态和同批双向成交。M0 模块未改写。

手续费 2 bps/5 bps、固定请求/响应延迟 1 ms/0.5 ms 都是显式机制假设，历史合约规则沿用 M0 的未认证状态。当前报告是 **交易 PnL**，不包含资金费率、保证金与强平；未将这些缺失项视为真实零成本。回撤按决策点和成交点观察，非每条 L2 更新盯市。

## 阶段边界

| 阶段 | 范围 | 当前状态 |
| --- | --- | --- |
| M0 | 引擎、双时钟回放、全日数据与性能 | 已完成 |
| M1 | 固定报价、原生订单、库存、交易账本、A/B/C 机制实验 | 本次交付 |
| M2 | 订单簿特征、短期 Alpha、预测与动态报价 | 未开发 |
| M3 | 逆向选择保护、质量归因、延迟/队列敏感性、C++ 性能优化 | 未开发 |
