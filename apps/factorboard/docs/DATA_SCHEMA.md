# FactorBoard 数据字段与版本规范

以下字段来自当前 `market_groups_v2` 研究数据及后端输出约定。必要字段应以服务器原有 Parquet schema 为准，DataFrame 在 `timestamp_ns` 上严格升序、唯一，样本间隔通常为 100ms。

## 1. 输入列与来源

| 字段 | 含义与单位 | 来源 / 有效性 |
|---|---|---|
| `timestamp_ns` | 当前样本本地可见/模拟接收时刻，UTC 纳秒 | 必需；另一可接受名称是 `ts_local_ns`、`local_ts_ns` 或 `sample_ts_local_ns` |
| `date` | 文件名中的 `YYYY-MM-DD` 日期 | 每个文件唯一，作为训练/验证/测试按日切分键 |
| `mid_price` | 当前最优买卖价平均值，USDT/BTC | 用于缺失预计算标签时严格推导未来收益 |
| `book_segment_id` | 盘口连续质量区段编号 | 不允许窗口和推导标签跨区段 |
| `trade_segment_id` | 主动成交连续质量区段编号 | 不允许成交滚动窗口跨区段 |
| `book_feature_valid` / `book_valid` | 当前盘口/盘口特征已通过数据质量校验 | 盘口因子必需时应筛掉假值和缺口 |
| `trade_feature_valid` / `trade_valid` | 成交特征已通过有效性检查 | 成交因子必需时应筛掉假值和缺口 |
| `label_valid` / `target_valid` | 未来收益标签的有效性标志 | 标签列若携带这些字段，就只使用有效样本 |
| `imbalance_1/5/10` | 前1、5、10档的挂单量失衡，无量纲 | `book` Features；正值通常表示买量相对更大 |
| `trade_imbalance_100ms/1s` | 最近100ms或1s的主动成交买卖量失衡，无量纲 | `trades` Features；已知零成交可为0，未知不能补零 |
| `bid_depth_10` / `ask_depth_10` | 当前买/卖前十档的显示挂单量，BTC | `book` Features |
| `spread_ticks` | 当前最优买卖价差，Tick | `book` Features |
| `microprice_minus_mid` | 微价格与当前中间价之差，USDT/BTC | `book` Features |
| `buy_volume_1s` / `sell_volume_1s` | 过去1s主动买入/卖出成交数量，BTC | `trades` Features |
| `buy_volume_5s` / `sell_volume_5s` | 过去5s主动买入/卖出成交数量，BTC | `trades` Features |
| `return_100ms_bps` / `return_1s_bps` / `return_5s_bps` | 截至当前时刻已经发生的过去收益，bps | `returns` Features；只能作为历史输入 |
| `y_mid_100ms_bps` | 未来100ms中间价收益，bps | 评价标签，不能用于构造当前 Factor |
| `y_mid_500ms_bps` | 未来500ms中间价收益，bps | 同上 |
| `y_mid_1s_bps` | 未来1s中间价收益，bps | 主标签；后端兼容 `target_1s_bps`、`label_1s_bps` 等名称 |
| `y_mid_5s_bps` | 未来5s中间价收益，bps | 辅助多尺度标签 |

若缺少某个收益标签、但 `mid_price` 与 `timestamp_ns` 完整，后端可以在**严格相邻100ms网格**上用未来相应时刻中间价生成标签。遇到时间缺口、跨盘口区段、缺失价格，样本置无效；不会向前或向后跨缺口插值。

## 2. 因子注册字段

| 字段 | 含义 |
|---|---|
| `id` | 全局唯一的语义英文因子名（snake_case），如 `book_obi_l5_mean_10` |
| `gen` | 所属研究批次，例如 `Gen001`；基础 Feature 使用 `FEATURES` |
| `category` | 市场信息类别：`book`、`trade`、`cross`、`returns` 等 |
| `desc` | 因子的简短经济含义或待检验假设 |
| `base_name` | 里层基础变量的语义名称 |
| `base_columns` | 里层计算直接依赖的输入 Feature 列名，1或2个 |
| `base_op` | 里层计算操作：`identity`（直接读取）、`difference`（两列相减）、`ratio`（第一列除第二列） |
| `operation` | 外层算子：`identity`（无）、`mean`（滚动均值）、`mean_diff`（快慢窗口均值之差） |
| `short_window` | 外层短窗口的100ms采样点数，例如10=1s |
| `long_window` | 长窗口100ms采样点数，例如50=5s，仅 `mean_diff` 需要 |
| `post` | 后处理名称，首版只实现 `raw`，即输出因子后不再归一化/截尾 |
| `raw_fields` | 此因子可追溯的上游原始盘口/成交字段说明；不作为动态计算代码 |
| `unit` | 输出单位，如无量纲 `ratio`、`bps`、`BTC` |
| `status` | 研究状态：候选、观察、已入选、已淘汰 |
| `notes` | 可编辑研究笔记，保存到 SQLite |
| `updated_at` | 数据库最后更新时间，UTC ISO8601 |
| `summary` | 最近一次完成的评估指标摘要；未评估时为 `null` |

## 3. Evaluation 输出

| 字段 | 含义 |
|---|---|
| `factor_id` | 对应的因子 ID |
| `dates` | 此次评估涵盖的数据日期列表 |
| `split` | 每日训练/验证/测试归属（`train` / `valid` / `test`） |
| `scope` | `test` 表示有独立测试日；`train_exploration` 表示只有单日、仅作探索 |
| `factor_definition` | 生成此份评估所用的完整因子公式与输入列约定 |
| `data_manifest` | 本次读取的文件相对路径、大小、修改时间，用于源数据追踪 |
| `clock_spec` | 计算使用的时钟和样本间隔语义 |
| `group_method` | `quintile`：训练期固定五分位；`sign`：负/零/正三组 |
| `group_thresholds` | 只使用训练日数据估计的分位阈值 |
| `hist` | 测试期总体直方图桶中心 `x` 和占比 `v` |
| `pct` | 逐日 P10/P25/P50/P75/P90、零值率及样本有效率 |
| `densities` | 每日在统一训练期桶边界上的直方图占比；用于热图 |
| `days` | 每天主期限1s的 Pearson IC、Rank IC 和有效样本数 |
| `daily_by_horizon` | 按日、按预测期限的 IC、Rank IC、样本数和训练/验证/测试标识 |
| `hor` | 按期限汇总的测试日平均 IC、最高组−最低组 bps 差，及有效日期数 |
| `label_origins` | 各预测期限读取到的原始标签列名，或 `derived_mid_price`（以严格连续网格推导） |
| `groups` | 主期限1s各分组的逐日均值汇总、有效样本数量 |
| `groups_by_horizon` | 100ms/500ms/1s/5s 分组未来收益结果 |
| `series` | 一段局部时序采样，包含因子、未来1s标签、可用时则包含历史收益 |
| `summary.meanIC` | 测试日平均 Pearson 时序 IC；无有效测试日时为 `null` |
| `summary.rank` | 测试日平均 Spearman Rank IC（样本非常大时按固定步长抽样，并记录 `rank_stride`） |
| `summary.icir` | 测试日 IC 均值 / 测试日 IC 样本标准差；不年化 |
| `summary.positive` | 测试日 IC 为正的有效日期比例 |
| `summary.spread` | 未来1s最高因子组−最低因子组平均收益差，bps；不是可实现策略收益 |
| `summary.valid` | 计算有效因子值占所有读取样本的比例 |
| `summary.zero` | 有效因子值中数值等于零的比例 |
| `summary.n` | 具有有效 IC 的独立测试日期数 |
| `summary.row_count` | 此次读取的100ms研究样本总行数 |
| `summary.label_status` | 各预测期限是否有原始或推导后的有效标签来源 |

## 4. 后台任务

| 字段 | 含义 |
|---|---|
| `id` | 任务唯一标识 |
| `kind` | `evaluation`，表示计算并缓存因子评估 |
| `factor_id` | 目标因子名称 |
| `config` | 使用的数据日期等计算参数 |
| `status` | `queued`、`running`、`completed`、`failed`、`interrupted` |
| `progress` | 0到1之间的进度比例 |
| `error` | 失败时的异常说明，不作为研究结果 |
| `result_path` | 成功后的缓存结果 JSON 文件 |
| `created_at` / `updated_at` | 任务创建与最后更新时间（UTC ISO8601） |

任务结果由工作进程写临时文件后原子替换，SQLite 只存索引和主指标摘要。API 与研究计算分离，浏览器只请求统计结果而不接收完整原始数据。
