# 2026-11-06 — 高频研究数据与Gen001

当前Data使用market_v2标准行情，Features使用market_columns_v1，Factors使用gen001_v1。完整命令、字段及模块见[研究包指南](research_package.md)。

原始官方普通成交与25档快照按类型和日期标准化。100ms基础研究列以交易所时间加20ms固定延迟对齐，异常值设为null，正常零成交保持零。

每个Feature和Factor各自保存完整日期区间。滚动窗口按时间网格连续计算，跨午夜和缺失位置继续滑动，由min_samples控制最少有效观测。当前10点、50点窗口分别要求8和40点。基础列与衍生列都按变量命名Parquet，文件锁按变量独立设置。

Gen001计算一档、五档、十档盘口失衡，100ms及1s成交失衡，以及成交与盘口压力背离的短长窗口均值差。evaluation独立完成分布、逐日IC、多周期IC和分组收益。

独立标签继续按日期和预测周期保存，主单位bps，同时保存ticks及USDT。标签要求未来精确时点、同一UTC日及区间内连续有效盘口。当前结果报告位于 `workspace/results/evaluation/gen001_v1/report.html`。
