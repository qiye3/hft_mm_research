# 模型统一对照链路 2026-11-09

## YAML与输入输出

YAML是保存配置的文本文件。冒号左侧是参数名，右侧是参数值，缩进表示归属关系。本轮使用两份配置。

| 配置 | 所属阶段与执行入口 | 输入 | 输出 |
| --- | --- | --- | --- |
| `config/models/comparison.yaml` | 模型研究，`scripts/run_model_comparison.py` | 531天正式Features、1s Factors、五周期Labels | 训练期候选、共享磁盘矩阵、冻结模型、逐行预测和研究报告 |
| `config/research/model_comparison.yaml` | 服务器编排，`scripts/run_overnight_server.sh` | 上述模型配置和各阶段依赖 | 独立任务状态、日志、CPU和内存采样 |

当前结果版本为 `workspace/results/models/comparison_2026-11-09_v2/`，缓存为 `workspace/data/model_matrices/comparison_2026-11-09_v2/`。

## 共同研究口径

研究区间为2025-05-09至2026-10-21。训练截至2026-05-31，验证为2026-06-01至08-31，历史回看测试为2026-09-01至10-21。训练和验证末端分别剔除300秒，并检查标签目标时间未跨越分界。数据首部预热120秒，跨午夜持续读取历史。

候选列仅读取训练期的1s日均IC、有效天数和样本覆盖率。候选训练覆盖至少95%，有效IC日至少达到训练日数的90%。90列由训练排名80列和训练期挑出的10个家族代表构成。16列代表、最多40列深度扩展及12列序列输入均用训练期确定。家族按明确的构造组配置归属，保留实际清单。

所有模型使用同一批有效标签时间戳。各列缺失填训练均值，另附等宽缺失标记。标准化只拟合训练期，输入按同一规则clip到±20。正常化缓存使用Float32，正式Factors保持Float64，线性模型使用Float64充分统计。

共同训练候选中固定抽取600万行，全部模型使用同一行集。调参固定使用验证期均匀取出的10万行；参数冻结后计算完整验证与测试期。GRU每轮遍历同一训练行集，历史输入从完整网格读取，缺失位置保留。每个输入组的样本集合一致。

已有Test曾被前期探索使用，本轮报告明确标记为历史回看。模型方向、选列与参数继续遵守本轮训练和验证边界，后续新封存数据承担新的泛化验收。

## 内存与跨日处理

1. 每次最多预读4个日期，只读行情输入和标签。主线程按日期顺序构造派生量、累积训练统计并顺序写入NPY。
2. 跨日携带120秒原始整秒输入，计算1/5/30秒滞后和30/120秒均值偏离。缺失位置继续参与窗口位置计数，均值至少80%有效。
3. 深度比率使用双侧深度变化差，除以双侧变化绝对值和，分母下限1e-6。派生定义随模型输入清单管理。
4. 线性训练按25000行合并充分统计。LightGBM分块生成只读训练映射，使用相同的训练均值填补和缺失通道。GRU按批次读取过去序列。
5. 数据准备和树训练分别设置24GiB、28GiB匿名分配上限。GPU任务的CPU匿名分配上限12GiB，显存上限80%。全局另保留8GiB准入余量和6GiB紧急余量。
6. 单个训练或评价任务失败后保留日志，其他独立任务继续。冻结与报告列出未完成模型，禁止把缺少的结果写成正常指标。

## 固定对照实验

输入组包括历史价格、历史加盘口压力与净深度、16列家族代表、90列基础池、深度扩展、序列扩展、全量输入、去深度组和去OBI组。去组实验按构造家族定义，其他组可以保留同一原始盘口的其他表达，结果解释为当前模型中的条件贡献。

Ridge覆盖全部输入组，OLS与Lasso比较90列和全量输入。LightGBM覆盖全部输入组，叶子数31/127由共同验证位置选取，最多1200轮。GRU使用同一90列，分别观察1/30/120秒历史。

所有候选调参结束后先保存冻结清单，再评价测试期。分组阈值取共同训练子样本，验证与测试使用相同阈值。主模型预测1s，其余10/60/180/300s只用于信号衰减评价。模型表IC按完整区间计算，增量按同日IC差进行配对Bootstrap，两个模型共用抽样日期。

## 执行与检查

```bash
bash scripts/run_overnight_server.sh config/research/model_comparison.yaml
```

`execution/status.json`记录独立任务状态，`execution/resources.jsonl`记录资源，`execution/任务名/run.log`记录进度。`selection.json`、`rows/`、`contract.json`、`models/`、`frozen.json`、`evaluation/`和`report.md`记录完整研究过程。`source_snapshot.tar.gz`保留实际执行源码与配置。正式启动依赖GNU time，新容器先安装系统包`time`。

## 本次文件职责

| 文件 | 改动 |
| --- | --- |
| `research/mm_research/models/comparison_contract.py` | 训练期选列、家族定义、派生表达式和共同时间位置 |
| `research/mm_research/models/experiment_data.py` | 有界日期预读、跨日派生、顺序磁盘缓存，共享缺失变换 |
| `research/mm_research/models/experiments.py` | 复用移至数据模块的历史读取函数 |
| `research/mm_research/models/comparison.py` | 共同训练样本上的线性、树、GRU训练及冻结与评价 |
| `research/mm_research/models/comparison_metrics.py` | 分块预测、五周期指标、训练阈值分组及配对日级Bootstrap |
| `research/mm_research/models/comparison_report.py` | 统一研究报告及缺失实验记录 |
| `research/mm_research/config.py` | 新增严格模型对照配置字段 |
| `research/mm_research/overnight.py` | 可关闭定时报告，指定本轮报告入口 |
| `scripts/run_model_comparison.py` | 明确的分阶段入口 |
| `config/models/comparison.yaml` | 数据口径、选列、样本与模型参数 |
| `config/research/model_comparison.yaml` | 有依赖关系的资源受控任务队列 |
| `tests/test_model_comparison.py` | 训练期选列、跨日、串并行一致性、缺失和配对统计回归 |
| `pyproject.toml` | 明确模型环境的scikit-learn和threadpoolctl依赖 |
| `AGENTS.md`与架构批注 | 记录长期规则与本轮链路，本轮执行快照核心编号1—204 |

其他文件中的核心编号仅作统一顺序更新。既有工作区修改按原内容保留。

## 2026-11-09完成结果

25个模型的共同样本评价已全部完成。冻结模型为完整199列LightGBM，验证IC 0.24265，历史测试IC 0.25029、R² 6.239%。深度与历史派生合并相对90列基线，日均IC提高0.00754，91/92个验证日改善。配对报告同时给出IC差和MSE减少量，去组实验对照完整输入模型。

数据准备峰值匿名内存3.15GiB、RSS 23.32GiB；全任务最大RSS 43.86GiB，文件映射页与匿名分配分开报告。本轮无新增OOM。全链路约115.3分钟，完整结果见[统一研究报告](../reports/model_comparison_2026-11-09/report.md)。实际执行与最终报告代码分别归档为source_snapshot.tar.gz和report_source_snapshot.tar.gz。
