# 远端因子评测方法、结果输出与看板展示

本文档梳理 qiye-rtx3090-10vcpu 服务器上当前生产因子评测的完整口径：上游研究评测流水线（方法与输出文件）、看板只读适配层（接口与列表指标）、以及前端页面实际展示的内容。对应代码为 `research/mm_research/evaluation/`（评测）与 `apps/factorboard/`（看板）。

## 一、上游生产评测流水线

入口 `scripts/evaluate_factors.py`，配置在 `config/evaluation/`。当前服务器已完成四个批次：

| 评估批次（output 目录） | 因子数 | 因子来源版本 |
| --- | --- | --- |
| `workspace/results/evaluation/server_factors_1s_v1` | 300 | 首批非OFI因子（Gen001—005口径） |
| `workspace/results/evaluation/server_factors_extra_1s_v1` | 360 | Gen011—Gen020 |
| `workspace/results/evaluation/server_factors_latest_1s_v1` | 252 | Gen031—Gen037 |
| `workspace/results/evaluation/server_factors_more_1s_v1` | 324 | Gen021—Gen029 |
| 合计 | 1236 | |

### 1. 输入与数据范围

- 因子：`workspace/data/factors/<版本>/<因子名>.parquet`，一个因子一个全日期文件。整秒输出、Float64，外层窗口按完整100ms输入计算、默认80%有效观测。
- 标签：`workspace/data/labels/server_mid_1s_v1/horizon={1s,10s,60s,180s,300s}/<日期>.parquet`，目标为未来中间价变化 `y_mid_<周期>_bps`，附带 `label_valid` 质量标记，支持跨日。
- 日期范围：2025-05-09 至 2026-10-21，共531个UTC交易日。来源切换2025-10-01构成标签质量边界。

### 2. 样本切分与隔离（dataset.py）

- 切分按日期：`train`（≤2026-05-31）、`validation`（≤2026-08-31）、`test`（其余，截至2026-10-21）。
- 训练/验证末日统一剔除末尾300s（`purge_ns`），并要求标签结束时间 `label_end_timestamp_ns` 不跨入下一分区，防止泄漏。
- 测试期不做剔除，由标签可用性决定样本范围。

### 3. 评估方法（metrics.py / report.py）

每个因子在每一天、每个预测周期上计算：

- **IC（Pearson）与 Rank IC（Spearman）**：仅在因子与标签共同有效的配对样本上；配对不足两条、因子或标签为常数时明确给出原因（`insufficient_pairs` / `constant_factor` / `constant_target`），不返回伪值。
- **采样方式**：`all`（全部整秒）与 `nonoverlap`（按周期非重叠采样，仅取 `timestamp_ns % horizon == 0` 的网格）。两种采样都保存。
- **分组收益**：分组边界只用训练期因子值拟合后冻结（`fit_group_boundaries`，【核心逻辑113】）：
  - 默认5组，边界为训练期精确分位数（线性插值），全程不重拟合；
  - 若分位边界重复（因子大量取相同值），自动退化为负/零/正三组（`sign` 模式），并在 `group_boundaries.json` 里显式记录；
  - 每组统计样本数、因子均值、条件未来收益均值（bps），并记录实际组数与请求组数，空组保留缺失。
- **高低组差 `top_bottom_bps`**：只在实际组数等于请求组数的日期，取最高组与最低组日均收益之差，再按日期等权平均；端点组缺失的日子不计入。
- **分布统计**（每日）：rows、valid_rows、missing_rate、zero_rate（分母为有效因子值）、mean/std/min/max、q01/q05/median/q95/q99、唯一值数。另累计全期直方图（训练期范围80桶，超范围归入两端）。

### 4. 调度与发布

- 按因子并行（`workers: 5`，批大小6），同批因子共享每日标签读取；源因子只读并持有共享锁。
- 每个因子独立发布：全部统计表写完（zstd压缩Parquet）→ 绘诊断图 → 最后原子写 `completed.json`（身份=评测配置+源因子定义）。已有相同身份的 `completed.json` 时直接复用，不重算。
- 发布期间持有独占评估锁，看板读取时共享同一把锁，不会读到半成品。

### 5. 单因子输出目录（`workspace/results/evaluation/<批次>/<因子名>/`）

| 文件 | 内容 |
| --- | --- |
| `summary.parquet` | 按 split × horizon × sampling 汇总：valid_ic_days、ic_mean/median/std、rank_ic_mean、positive_ic_day_ratio、paired_rows、top_bottom_bps、complete_group_days |
| `daily_ic.parquet` | 逐日：因子、日期、split、horizon_ns、sampling、rows/factor_valid_rows/label_valid_rows/paired_rows、ic、rank_ic、失效原因 |
| `daily_distribution.parquet` | 逐日因子分布统计（上节"分布统计"全部列）+ unique_values |
| `group_returns.parquet` | 逐日×周期×采样×组：samples、factor_mean、mean_return_bps、actual_groups、requested_groups |
| `group_boundaries.json` | 分组模式（quantile/sign）、冻结阈值、训练期min/max、train_end |
| `diagnostics.png` | 6宫格诊断图：全期直方图、逐日IC(1s)、各split多周期IC、测试期1s分组收益、日有效率、日分位曲线 |
| `report.html` | 单因子说明页，嵌入诊断图 |
| `completed.json` | 完成标记与身份（config + 源定义 + definition_version: one_second_five_horizons_v3） |

批次根目录另有汇总 `summary.parquet`（全因子拼接）和入口 `report.html`（因子链接列表）。

## 二、看板只读适配层（apps/factorboard/backend/project_board/production.py）

服务器看板**不重新计算任何研究指标**，只读取上述已发布结果：

- 服务入口 `scripts/serve_results.py`，由 Supervisor 常驻管理，仅监听服务器回环 127.0.0.1:8765，本机经 `deploy/connect.sh` SSH 隧道访问。Polars两线程、OpenBLAS单线程、单进程串行响应。
- 评估批次清单来自 `deploy/results.json`（上表四个根目录）；每60秒刷新一次索引，只收录有 `completed.json` 的因子目录，重名因子直接报错。新增批次需加入清单并重启。
- 读取时共享上游 `evaluation.lock`，避免读到发布中的混合状态；JSON 输出把 NaN 统一转 null。

### HTTP 接口

| 路由 | 说明 |
| --- | --- |
| `GET /api/v1/config` | 清单：日期范围、因子ID、scopes（train/validation/test）、horizons（1/10/60/180/300s）、列表固定 test+1s |
| `GET /api/v1/factors` | 全部因子列表及列表指标（见下） |
| `GET /api/v1/health` | 健康检查（只读标记、因子数） |
| `GET /api/v1/evaluations/<因子名>?scope&horizon&sampling` | 单因子详情，参数在HTTP边界校验（非法值422，未知因子404） |

生产入口在HTTP层把 `js/data.js`→`production_data.js`、`js/evaluation.js`→`production_evaluation.js`，其余静态文件原样提供；本机预览入口（十二组研究视图）保持独立，互不影响。

### 列表指标（固定测试期 1s、全部整秒）

- `ic` = ic_mean，`rank_ic` = rank_ic_mean，`spread` = top_bottom_bps
- `icir` = ic_mean / ic_std
- `positive` = 正IC日比例，`days` = 有效IC日数
- `valid` = paired_rows / 测试期总行数（共同覆盖率）
- `zero_rate`：按测试期各日 valid_rows 加权的零值率

### 详情 payload（单因子）

`summary`（当前选项行）、`daily_ic`（逐日IC/RankIC及计数）、`groups`（按组日期等权聚合的bps/样本/有效日）、`boundaries`（冻结阈值）、`daily_groups`（逐日×组）、`daily_horizons`（当前范围×全部周期逐日IC）、`split_summary`（当前周期×采样下三个split的汇总）、`distribution`（当前范围日分布）、`horizons`（当前范围各周期汇总）、`config`、`definition`、`definition_version`。

## 三、看板前端展示

### 1. 外壳（资料库组织，保留原Demo体系）

- 顶部工作区选项卡：**资料库 / 因子对比 / Gen 批次 / 评估任务**。
- 资料库三栏：左侧分类（全部/Factors/Features/收藏/对比组/研究状态/自定义分类/Gen批次）、中间因子列表（可排序列、多选、搜索、筛选方案）、右侧详情。
- 浏览器本地保存收藏、研究状态、自定义分类、笔记、筛选方案和列设置，不写研究数据。
- 列表与筛选固定使用测试期1s全部整秒指标；详情内可切换范围/周期/采样。

### 2. 评估设置（详情上方可折叠）

范围（训练/验证/测试）、预测周期（1s—300s）、采样（全部整秒/非重叠标签）、方向（高多低空/高空低多）、是否扣除交易成本、单边费率bps（默认5）。这些只影响展示聚合，不写研究配置。

### 3. 详情标签页

- **评估总览**：四个指标卡（日均IC、日均Rank IC、共同覆盖率、双侧等权收益）+ 笔记 + 四张图：每日预测相关性（IC/Rank IC折线）、分组未来价格变化（柱状，训练期冻结五组或负零正三组）、预测周期与信号强弱（各周期日均IC）、每日共同覆盖率。
- **数据与分布**：每日因子分位数（P5/P50/P95）、缺失率与零值率曲线，及全列分布表。
- **时序**：展示已有评测保存的日级因子统计（每日均值、每日标准差）。当前评估未保存秒级价格联动，不做秒级曲线。
- **IC与衰减**：逐日IC/Rank IC、多周期高低组收益差，及周期汇总表。
- **分组收益**：原始分组收益柱状图、分组样本分布柱状图、完整分组表（组别/日均收益/总样本/有效日）。原始分组收益永远为毛收益。
- **多维诊断**：四个视角，每个视角四张图（详见下节）。
- **口径与版本**：评估口径文字说明 + 完整config/定义/分组阈值JSON。

### 4. 多维诊断（production_diagnostics.js，纯前端派生）

- **稳定性**：日IC直方图（有限值、6个等宽区间、常数序列单独一组）、七日滚动IC（真实日历窗口、至少5个有效日、不年化）、月度IC/Rank IC（月内有效日等权）、月度正IC日比例（零IC归入非正日）。
- **预测周期**：月份×周期热图（各月日均IC、对称配色、悬浮显示有效天数）、训练/验证/测试三段对比（当前周期与采样口径）、多周期预测能力曲线、周期与高低组收益差（毛收益）。
- **分组结构**：组别与收益方向（原始bps柱状）、组别样本占比（分母为所有组样本合计）、相邻组收益变化（各组日均收益之差，有效日期可能不同）、逐日高低组差（仅当天两端点组同时存在，否则缺失）。
- **数据质量**：以日期关联因子分布与当前周期日IC——共同覆盖率×IC、缺失率×IC、零值率×IC三张散点（每点一天；三者的分母口径不同）+ 覆盖/缺失/零值日度曲线。

### 5. 因子对比

- 不限数量，弹窗按名称/Gen/分类搜索选择。
- 对称条形图对比各因子日均IC与日均Rank IC（完整ID悬停查看）。
- 两张散点：IC日间波动（日IC样本标准差，不年化）×日均IC、共同覆盖率×日均IC。
- 对比表：因子、IC、Rank IC、高低组差bps、双侧等权情景bps、共同样本。

### 6. 成本情景口径

总览与对比表中的"双侧等权收益"计算方式：

```
净收益 = 0.5 × 方向 × 高低组差(top_bottom_bps) − 2 × 单边费率
```

即高低组差对半拆到两侧持仓、扣往返手续费；默认单边5bps（来自项目基线taker费率0.0005），可调，不自动选方向。这是条件收益情景，**不是**订单成交后的策略PnL；原始分组收益图永不扣费。策略级成本模拟另属 `evaluation/trading.py` 链路，未接入本看板。

## 四、本机预览与生产视图的关系

- 本机 `project_preview.py prepare + start.sh` 打开的是**研究预览视图**：基于 gen001 六因子（2026-07-01至07-30）独立计算十二组（G0.1/G1—G10/G9.9，两尾与端点重叠）、逐日/固定双阈值、延迟诊断（0—1000ms）、时序查询（6000个100ms网格）和六因子相关矩阵，产物在 `workspace/results/factorboard/runs/`，含三日移动块bootstrap区间（2000次、固定种子）。
- 服务器生产视图只展示已发布评估（本文第一、二节口径），不包含十二组、延迟诊断和秒级时序；两者共用资料库外壳与浏览器偏好，评估入口由服务端显式切换。
