# 原生行情下载恢复

2026-11-14，3月24日原档下载失败已定位为容器DNS搜索域问题。默认 `ndots:5` 使 `pan.baidu.com` 被解析到 `pixie.porkbun.com` 对应地址，TLS握手失败；使用进程环境 `RES_OPTIONS=ndots:1` 后正常解析到百度地址，HTTPS恢复。

独立预取进程已启动，观察到临时文件从0增长到368MiB，尚未标记完整下载。继续使用证书验证和服务器网络出口，不改全局DNS。

恢复命令：`RES_OPTIONS=ndots:1 PYTHONPATH=research .venv-models/bin/python scripts/run_cost_native.py config/strategy/gen041_native_risk_20261114.yaml prefetch`。

尝试修改等待中的队列任务环境触发了监督器配置保护。已恢复原定义并确认与保存状态一致；原生回放子任务继续运行。后续只在独立预取进程设置环境，不修改运行中的队列定义。
