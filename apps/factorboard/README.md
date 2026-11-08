# BTCUSDT FactorBoard

独立的高频因子研究看板。只读项目已生成的因子、标签和评估，提供十二组诊断、成本情景、时序查询和多日期比较；不修改主项目的计算代码或五分组结果。

## Quickstart

在项目根目录执行：

```bash
# 原始研究结果更新后执行；仅在看板自己的结果目录生成诊断。
.venv/bin/python apps/factorboard/scripts/project_preview.py prepare

# 打开 http://127.0.0.1:8765；Ctrl+C停止。
apps/factorboard/scripts/start.sh
```

可传入 `--port 8766`。使用现有环境中的 Polars、NumPy，不需要安装原 Demo FastAPI 依赖。

详见 [计算口径与修改清单](docs/PROJECT_PREVIEW.md)。当前正式入口是 `project_preview.py`；原 ZIP 的 `factorboard_api/`、Docker/Compose、合成样本及其测试保留作参考，不是当前项目看板的运行路径。

## 服务器部署

服务器版本读取已有完成评测，覆盖531日、840个因子。使用方式、评估口径和服务管理见 [服务器部署说明](docs/server_deployment.md)。

```bash
bash apps/factorboard/deploy/connect.sh
```

保持连接后打开 http://127.0.0.1:8765/ 。生产评估清单在 `deploy/results.json`。
