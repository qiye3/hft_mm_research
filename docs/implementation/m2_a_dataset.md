# 2026-11-06 — M2-A研究数据集

当前生产统一使用market_v2标准行情，完整流程和命令见 [研究包指南](research_package.md)。

官方普通成交和网盘25档快照先生成两份标准Parquet。Polars以交易所时间加20ms固定延迟建立100ms网格，再生成独立版本的特征与多周期标签。

每个完整UTC日保留864,000个网格点。base特征组保存前十档盘口、Mid、Spread、成交笔数、主动买卖量与金额、模拟可见时间及独立盘口和成交质量字段。上游标准快照完整保留25档，研究可按需要扩展读取深度。

正式快照研究选择16个特征，分为盘口、主动成交与历史收益三类。精确事件OFI使用L2事件路径，公式和C++提取能力保留在现有模块。

标准文件按 `standard/market_v2/{snapshots,trades}/YYYY-MM-DD.parquet` 保存，特征按 `features/market_groups_v2/YYYY-MM-DD.parquet` 保存，标签按 `labels/market_mid_groups_v2/horizon=周期/YYYY-MM-DD.parquet` 保存。base列与衍生特征共同位于每日特征文件。

未来标签支持100ms、500ms、1s和5s，使用精确未来网格的Mid计算bps、Ticks及USDT三种单位。窗口内盘口必须连续有效且位于同日，日末没有未来价格时标签为空。

book组检查当前盘口，trades组检查5秒成交历史，returns组检查5秒盘口历史。读取按所选组的独立有效标记筛选。

普通成交编号空缺保持原样，真实字段异常与盘口中断按质量字段处理。历史滚动窗口只使用采样时刻以前的数据，特征和标签分别判断有效性。

旧研究入口和缓存已清理。历史实验报告保留其原日期和测量口径，当前分组结果见`workspace/results/m2_standard/market_groups_v2/`。
