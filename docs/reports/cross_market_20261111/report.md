# 跨市场180秒直接模型实验｜2026-11-11

本批直接比较新增信息对原146列模型的贡献。数据读取和20列变量计算并入实验，不建立新的标准行情生产阶段。最终成绩由服务器结果目录中的 `summary/models.csv` 与 `summary/report.md`给出。

| 新信息组 | 新列数 | 模型 |
| --- | ---: | --- |
| spot：现货压力、现货收益、现货收盘价与永续Mid偏离、两市场收益差 | 16 | XGBoost seed17；MLP seed17/42/73 |
| premium：溢价水平和60/180/600秒变化 | 4 | 同上 |
| all：两组组合 | 20 | 同上 |

普通窗口为10/60/180/600秒。现货秒线在结束加20ms之后的下一个整秒使用，永续Mid取已有市场矩阵上一整秒；分钟溢价同样只读取已结束分钟。输入时间键、拟合行、内部早停行和180秒对数bps目标沿用旧模型。训练截至2026-03-22，开发截至05-14，历史复核截至07-06。旧模型权重与成绩直接复用。

输出总体IC、Rank IC、R²、逐日配对IC增量、正增量日期比例和日期重采样区间。第一批区间为描述性统计，多组最优结果还需后续固定比较，不直接据此宣称盈利。

YAML为参数配置。`config/models/cross_market_20261111.yaml`由 `scripts/run_cross_market.py`读取，输入已有矩阵与官方月档，输出新增变量缓存、12个模型和预测；`config/research/cross_market_20261111.yaml`由既有后台监督器读取，安排输入、训练和报告任务，失败节点最多重试两次，其余模型继续。

服务器在独立源码快照运行：

```bash
cd /root/shared-nvme/hft_mm_research/workspace/results/cross_market_20261111_v1/source_snapshot
PYTHONPATH=research .venv-models/bin/python -u scripts/run_research_background.py config/research/cross_market_20261111.yaml
```

结果目录为 `workspace/results/cross_market_20261111_v1/`。队列状态在 `queue/status.json`，逐项日志在各任务目录。初次启动补齐了既有队列要求的目录字段；旧输入缓存因实例临时盘配额不足迁至共享盘，fit/tune直接引用原矩阵。修复不改变模型和样本，记录保存在 `storage_repair.json`及任务日志中。

核心公式位于 `research/mm_research/factors/cross_market.py`；实验编排与缓存位于 `research/mm_research/models/cross_market.py`；训练复用 `models/input_comparison.py`及原GPU模型。既有数据、第三方引擎和已完成模型不修改。

## 本轮结果

12个模型全部完成。现货组XGBoost的开发IC为0.03715（旧0.03095），历史复核IC为0.02783（旧0.01814）。开发期日均IC增量0.00500的日期95%区间为[-0.00077,0.01046]，复核增量0.01151的区间为[0.00465,0.01939]。现货组存在值得继续拆分的预测增量，单独溢价及MLP各种子表现较分散。下一轮优先区分成交压力和价格联动的贡献。

[完整模型成绩](/Users/qiye/Desktop/docs/Projects/HFT_MM/Results/research_180s_optimization_20261111/03_跨市场模型实验结果_20261111.md)。本轮未运行新信号的交易回放。
