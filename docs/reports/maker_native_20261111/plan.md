# 原生Maker成本机制实验｜2026-11-12

此前固定本金账户实验没有在开发和冻结复核两段同时达到30%年化。本轮改变真实执行方式，原信号、费用表和训练模型保持不变。先只运行开发区间2026-03-23—05-14，完整53天计入年化；开发未达标时不查看此规则的复核收益。

## 固定策略规则

- 本金10000USDT，每次目标0.10BTC，最高实际总敞口1倍。
- 信号来自已经冻结的seed17、30分钟费用分类器。空仓且信号非零时，按方向挂买一或卖一，使用原生risk_adverse队列模型与Post-only订单。
- 首次实际成交后开始持仓计时，撤销尚未成交的入场剩余数量；取消在途期间的成交仍归实际仓位。
- 持仓7200秒后开始退出：在对手方向的一档挂Maker单，最多等待30秒。部分退出后继续处理剩余实际仓位；超时先撤销Maker单并等待确认，再逐批IOC平仓。
- 30秒同时为本策略的单笔挂单TTL和退出排队总期限。退出期间重新报价不能延长总期限。区间末端保留20秒强制撤单和IOC退出，不再等待Maker。
- 实际Maker成交收费2bps，实际Taker成交收费5bps；资金费按结算时实际库存计入。不能把未成交当成交，也不能仅替换旧账本费率。

## 实现和验收

C++新增directional_maker模式，使用已有MarketMaker、Backtest和账本。核心393仅负责实际残仓的Maker减仓。合成原生事件分别覆盖Maker开平、Maker入场但退出超时、部分Maker退出后IOC、零信号不入场。

Python原生编排显式配置mode、order_quantity、ttl_ms和splits；已有原档按明确大小完整性复用，无需再次读取网盘凭证。历史实验保持其原source_snapshot可复现，当前配置新增明确字段后不要覆盖旧冻结输出。

核心394在全部区间结束后复用原生连续权益核算，补充完整日历年化、费用后回撤、峰值总敞口、周/月收益和Maker/Taker成交金额。开发门槛为年化超过30%、峰值敞口不超过1倍、去掉最好周仍盈利；跨月集中度另行分析，不因门槛通过就宣称稳定实盘盈利。

只有开发通过才冻结这套规则并执行后续历史复核。先保存全部开发结果，包括无成交和亏损区间。

配置：config/strategy/maker_native_20261111.yaml。后台：config/research/maker_native_20261111.yaml。入口仍为scripts/run_cost_native.py。产物：workspace/results/maker_native_20261111_v1。事件转换与回放交错，最多提前准备一个区间，使用已缓存原始L2和官方成交。

## 本轮启动记录

已完成原生编译和执行模式测试。新账户汇总在独立reference_audit目录复算旧账本，开发净收益5.539285USDT、复核14.990675USDT，与旧结果一致。开发计划为7个区间、34个必要事件日，其中8日复用已有事件缓存，其余复用原档。后台队列已启动，结果尚未完成。

重现时在独立source_snapshot中依次执行：

```bash
PYTHONPATH=research .venv-models/bin/python scripts/run_cost_native.py config/strategy/maker_native_20261111.yaml build
PYTHONPATH=research .venv-models/bin/python scripts/run_research_background.py config/research/maker_native_20261111.yaml
```

第一条命令完成Release构建及执行模式测试，第二条接管数据准备、回放和完整账户报告。队列的build任务只确认已构建程序存在。
