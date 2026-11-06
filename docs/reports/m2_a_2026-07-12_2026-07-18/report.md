# 2026-11-05 — M2-A 七天 BTCUSDT 研究数据集验收

> 版本说明。本文记录七天同源CSV与C++回放的历史验收，时钟和质量规则对应当时的m2_a_v1版本。当前生产使用官方普通成交、标准Parquet、固定20ms延迟和分组特征，操作入口见[研究包指南](../../implementation/research_package.md)。


## 一、交付概况

本次使用百度网盘 2026-07-12 至 07-18 的同源增量盘口和逐笔成交，通过已有 C++ 原生引擎回放生成研究数据。网盘 25 档快照用于独立表示核对，生产状态始终来自全深度增量回放。

七天共回放 792,387,216 条逻辑事件，其中逐笔成交 19,241,329 条。每日以 UTC 固定 100ms 网格生成 864,000 行，合计 6,048,000 行。经过盘口有效性、成交序号疑点、5s 预热和未来 1s 标签检查，保留 4,371,988 条可用研究样本。

研究表包含 19 个基础特征和未来 1s Mid 变化标签，所有日期均保留基础盘口表、研究表与质量标记。已实际通过 Python 读取完整七天，得到 `X.shape=(4371988, 19)`、`y.shape=(4371988,)`，特征与标签均为有限数值，日期时间索引唯一。

## 二、逐日结果

| 日期 | 逻辑事件 | 有效盘口样本 | 可用研究样本 | 成交序号疑点行 |
| --- | ---: | ---: | ---: | ---: |
| 2026-07-12 | 91,611,358 | 863,992 | 705,993 | 3,129 |
| 2026-07-13 | 144,414,669 | 863,851 | 600,105 | 6,129 |
| 2026-07-14 | 121,209,968 | 845,687 | 559,306 | 7,104 |
| 2026-07-15 | 117,194,257 | 863,960 | 627,752 | 5,243 |
| 2026-07-16 | 120,505,508 | 863,816 | 640,050 | 4,755 |
| 2026-07-17 | 127,770,033 | 863,938 | 579,241 | 6,412 |
| 2026-07-18 | 69,681,423 | 863,967 | 659,541 | 4,591 |

合计 6,029,211 个有效盘口样本与独立快照逐档一致。原始成交独立分桶与 C++ 的买卖量、金额、笔数一致。七天原始成交均未发现重复编号和非正价格、数量。

实际行情接收时间在七天转换中均未平移。C++ 小型夹具检查了网格边界、同时间戳跨分片、OFI、深档晋升与快照恢复。Python 检查了历史窗口、未来标签、因果质量标记与发布校验，第一天研究表重新构造后与保存内容逐项一致。

## 三、数据问题与处理决定

7 月 14 日有明显的盘口更新中断。UTC 12:30:08.8 至 12:52:26.2 的连续 13,375 个采样点因盘口过期被排除；附近还有其他较短中断。该日共 18,313 行盘口无效。七天总计 18,789 行盘口无效，包含日初尚无完整快照的样本。完整区间记录在各日 `validation.json` 的 `book_gap_summary` 中。

成交编号跳跃作为疑点保留。序号跳跃可能涉及特殊交易记录或采集问题，本次采用保守规则，在下一笔成交被收到、跳号首次可见时标记该区间，随后重新积累 5s 历史。这个决定显著减少了可用样本，实际模型研究应沿用同一质量规则，或在取得来源证据后另建数据版本。

默认盘口最大年龄为 1s，窗口预热为 5s。标签为同日同一研究区段的 `Mid(t+1s)-Mid(t)`，预测区间跨快照、过期盘口或成交序号疑点时无效。每日最后十行不生成标签。清洗后的 X 只使用清单声明的历史特征。

两套来源约 122ms 的接收时间差已经独立诊断。正式生产使用同源原始接收时间，保留来源记录；具体时钟与传输贡献需要采集日志进一步归因。 [接收时间差报告](../time_alignment_2026-08-01/report.md)

## 四、容量与保存位置

以下容量按十进制计算，分别表示不同内容。

| 内容 | 大小 |
| --- | ---: |
| 七天原始盘口与成交压缩文件 | 4.159 GB |
| 双侧路由标准化 NPZ | 9.942 GB |
| 基础与研究 Parquet 合计 | 887.60 MB |

研究数据位于 `workspace/data/research/m2_a_v1/`。总清单保存特征顺序、标签、日期和 `validated=true`。每个日期分区保存 `base.parquet`、`research.parquet` 和 `complete.json`。

校验、统计和交付汇总位于 `workspace/results/m2_a/m2_a_v1/`，包含每个日期的 `validation.json`、总体 `statistics.json` 与 `delivery.json`。 [数据集清单](../../../workspace/data/research/m2_a_v1/manifest.json) · [质量与数据量汇总](../../../workspace/results/m2_a/m2_a_v1/delivery.json) · [基础统计](../../../workspace/results/m2_a/m2_a_v1/statistics.json)

## 五、代码与复现

C++ 入口为 `extract_research`，`ResearchSampler` 在 Local 应用事件的前后接入固定网格与基础累计。Python 的 `build_research_dataset.py` 负责研究窗口、质量标记、标签和 Parquet；`validate_research_dataset.py` 负责独立核对；`summarize_research_dataset.py` 提供多日统计与 X、y 读取。

环境与 C++ 构建按 README 完成。原始数据已下载时，以下命令复用完成日期并核对发布记录。

```sh
python scripts/build_research_dataset.py config/research_dataset.yaml
python scripts/summarize_research_dataset.py workspace/data/research/m2_a_v1 \
  workspace/results/m2_a/m2_a_v1/statistics.json
```

重新生成数据使用新版本目录或显式 `--rebuild`。首次生产需要原始成交文件进行编号质检；后续因子研究读取 Parquet 即可。全部字段、时间边界和下载命令见数据集指南。 [M2-A 数据集指南](../../implementation/m2_a_dataset.md)

## 六、本组修改文件

| 分组 | 文件 | 具体变化 |
| --- | --- | --- |
| C++ 回放接入 | `src/mm/replay.hpp`、`src/mm/replay.cpp` | 增加可选的本地事件前后回调，供研究采样读取原生状态 |
| C++ 研究提取 | `src/mm/research_sampler.hpp`、`src/mm/research_sampler.cpp`、`src/apps/extract_research.cpp` | 固定时间网格、盘口、OFI、成交累计和 CSV 暂存入口 |
| 数据下载与诊断 | `scripts/download_baidu.py`、`scripts/diagnose_timestamps.py` | 可恢复下载、接收时间与交易所字段匹配、成交编号覆盖检查 |
| 研究表生产与读取 | `scripts/build_research_dataset.py`、`scripts/validate_research_dataset.py`、`scripts/summarize_research_dataset.py` | Polars 特征、因果质量标记、标签、校验发布、Parquet 和 X、y 读取 |
| 配置与验证 | `config/research_dataset.yaml`、`tests/test_research_dataset.py` | 七天参数与采样、OFI、质量标记、标签和发布校验测试 |
| 构建与依赖 | `CMakeLists.txt`、`requirements.txt`、`.gitignore` | 新目标与测试入口、固定 Polars/PyArrow 版本、忽略研究数据 |
| 项目文档 | `README.md`、`docs/reading_guide.md`、`docs/implementation/data_requirements.md` | 增加研究入口，说明已核实的交易所时间字段口径 |
| 阶段指南与报告 | `docs/implementation/m2_a_dataset.md`、`docs/reports/time_alignment_2026-08-01/report.md`、本报告 | 数据定义、操作方法、时间差证据和七天验收结果 |
