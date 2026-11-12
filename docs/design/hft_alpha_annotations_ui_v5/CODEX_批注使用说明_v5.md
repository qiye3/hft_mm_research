# 架构页面与批注使用说明 · 2026-11-11

## 当前版本

本页按2026-11-10本地代码更新。节点正文、输入输出与文件树描述当前已实现链路，各实验的运行完成度查看其workspace/results目录。

- 标准行情覆盖531天，Features按100ms计算，按单变量保存全日期Parquet。
- 当前四档Factors为10/60/180/600秒，共35组1206项；标签为七周期对数bps，主目标180秒。
- 1s统一模型对照、当前180秒GPU旧输入基线与新四档生产分别保存版本。
- C++实时线性模型与复杂模型冻结预测各有接入路径；原生订单、成交和账本复用第三方引擎。

维护源位于hft_mm_research/docs/design/hft_alpha_annotations_ui_v5/，同一套文件同步到外部hft_alpha_annotations_ui_v5/和服务器项目docs目录。每次更新同时维护HTML、批注JSON与定位JSON，保持节点ID和已有批注。历史批注保留当时上下文，当前实现查看节点正文。

## 用法

- 单击流程节点：查看右侧说明并高亮文件树。
- 双击流程节点、上方文件条目或右侧任意一条说明：打开独立批注编辑窗口。也可以点击右侧的「✎」入口。
- 批注可选择作者「我」或「Codex」，并标注「备注 / 问题 / 待办 / 已解决」，支持多条、修改和删除。
- 浏览器会尝试将批注保存至本地存储。请定期使用「导出批注 JSON」备份，换浏览器或更新 HTML 时可「导入批注 JSON」合并。

## 供 Codex 使用的数据文件

- `hft_alpha_annotation_targets_v5.json`：合法目标 ID 清单，包含上方文件路径、流程节点、右侧笔记分区及每一条笔记的 ID。
- `hft_alpha_annotations_v5.json`：批注记录。Codex 在任务结束后读取定位清单，追加或更新自己的记录，不覆盖其他批注。
- `hft_alpha_workflow_annotations_v5.html`：可交互页面。通过 HTTP 访问时自动加载同目录的批注 JSON，点击「同步 Codex 批注」可重新读取。

目标 ID 示例：

- `file:src/mm/features/feature_engine.cpp`：上方文件树的一个文件。
- `node:a2`：流程 A2 节点。
- `section:a2:处理`：A2 右侧「处理」分区。
- `note:a2:处理:0`：A2 右侧「处理」分区的第一条笔记；条目索引从 0 开始。

批注 JSON 示例：

```json
{
  "schema_version": 1,
  "annotations": [
    {
      "id": "codex-a2-feature-001",
      "target": "file:src/mm/features/feature_engine.cpp",
      "author": "Codex",
      "status": "待办",
      "body": "实现后补充增量 OFI 的数值一致性测试；保留测试命令与结果。",
      "created_at": "2026-11-04T12:00:00Z",
      "updated_at": "2026-11-04T12:00:00Z"
    }
  ]
}
```

`id` 是批注唯一标识。`target` 表示绑定的页面元素，必须从定位清单取值。`author` 是批注作者，`status` 是批注类型，`body` 是正文。`created_at`、`updated_at` 是 ISO 8601 时间，用于合并更新。同一批注后续修改时保持 `id` 不变，并更新 `updated_at`。

## 本地打开与同步方式

若双击打开本地 HTML（`file://`），浏览器通常不会允许网页自动读取旁边的 JSON。点击「同步 Codex 批注」或「导入批注 JSON」，选择 Codex 写出的 JSON 文件即可。

若希望修改 JSON 后自动刷新，可在 HTML 所在目录运行 `python -m http.server 8000`，用 `http://localhost:8000/hft_alpha_workflow_annotations_v5.html` 打开页面。此时刷新页面或点击「同步 Codex 批注」即可读取同目录 JSON。

如希望 Codex 直接修改一个完全独立的 HTML 文件，也可以更新 HTML 内 `id="codex-annotations-seed"` 的 JSON 内容。优先推荐使用独立 JSON 文件，避免修改界面脚本。

## 实验结果归档 · 2026-11-11

全部实验的参数、结果和逐项明细见[实验总档案](/Users/qiye/Desktop/docs/Projects/HFT_MM/Results/实验总档案_20261111/00_总览与阅读顺序.md)。最新信号策略、延迟与规模压力实验已结束，完成状态与研究结论已同步到core节点批注。

## 180秒可交易性研究 · 2026-11-11

三个冻结信号已完成尾部、跨月及成本研究。18个完整区间主动交易案例均为负，六日36例原生固定挂单完成成交归因。研究代码位于strategies/tradeability*.py，入口scripts/run_tradeability.py。研究参数和后台队列分别位于strategy与research配置目录。结果文档见[本轮研究结果](/Users/qiye/Desktop/docs/Projects/HFT_MM/Results/research_180s_tradeability_20261111/01_研究结果与结论_20261111.md)。本轮未保留候选，后续滚动训练及连续原生验证按筛选条件暂缓。

## 下一阶段研究计划与首批数据 · 2026-11-11

[180秒实施与持续优化计划](/Users/qiye/zxy_hft/hft_mm_research/docs/design/research_180s_optimization_20261111.md)规定数据、28列候选、模型对照、成本和订单研究的晋级规则。[首批跨市场样本](/Users/qiye/zxy_hft/hft_mm_research/docs/reports/cross_market_pilot_20261111/report.md)已核对完成。core节点记录待办计划，source节点记录已完成的样本核对；全历史生产和新模型尚未启动。

## 跨市场直接模型实验 · 2026-11-11

按用户要求取消独立适配与核对阶段，直接进行三组12个GPU模型对照。说明见[实验入口与分组](/Users/qiye/zxy_hft/hft_mm_research/docs/reports/cross_market_20261111/report.md)。core节点新增运行批注，样本核对笔记作为历史记录保留。

本批12模型已全部完成，现货组XGBoost在两段历史区间的IC均有提高，完整成绩已同步用户结果文档。后台队列正常结束，后续现货分组消融尚未启动。
