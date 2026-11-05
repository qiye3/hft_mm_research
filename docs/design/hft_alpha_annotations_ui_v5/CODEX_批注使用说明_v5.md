# 高频 Alpha 架构图 v5：批注与 Codex 同步

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
