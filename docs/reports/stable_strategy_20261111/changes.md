# 本轮修改范围

- `research/mm_research/models/net_value.py`与`run_net_value.py`：双方向费用后连续价值预测，比较稳健损失和仅训练期截尾；拟合与调参标签按预测周期隔离。
- `research/mm_research/strategies/stable_strategy.py`与`run_stable_strategy.py`：等权组合、跨周期共识、24个开发案例、完整周/月统计、开发冻结和历史复核。
- `research/mm_research/multihorizon/inventory.py`：在既有账本和逐档成交上增加信号失效退出、波动保护退出；管理使用决策可见报价，部分退出持续到真正平仓；旧策略分支不变。
- `research/mm_research/strategies/stability_attribution.py`与`run_stability_attribution.py`：按训练期冻结的波动分界及交易方向归因，保留未知环境，不改变选择规则。
- `config/models/`、`config/strategy/`、`config/research/`对应配置：分别定义训练输入输出、策略候选和后台资源依赖。
- `tests/test_stable_strategy.py`：覆盖旧规则一致性、部分退出、可见报价时钟、缺失风险尺度和滚动参考一致性。
- `AGENTS.md`和架构HTML/JSON批注：记录新链路及每轮较大修改验证后直接提交并推送的授权。

新增管理规则已在服务器现有环境完成直接相关验证。初次部署的macOS元数据和队列配置缺项已修复，原始失败日志保留，完整队列位于`queue_recovered`。成功模型复用完成标记，不重复训练。

项目原来未提交的跨市场、费用分类、原生回放与收益归因代码是本轮的直接研究依赖；相关原生结果见`docs/reports/cost_native_20261111/report.md`。实验数据、模型权重、下载凭证和大型结果文件均不纳入Git。
