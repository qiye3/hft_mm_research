#!/usr/bin/env bash
# 【非核心】在服务器独立会话运行常驻队列，监督进程异常退出后最多恢复三次。
# 输入夜间YAML路径，状态文件支持重新接管尚在运行的子任务，重复启动由文件锁拒绝。
set -u
config_file="$1"
attempt=0
while [ "$attempt" -lt 3 ]; do
    PYTHONPATH=research POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        .venv/bin/python scripts/run_overnight.py "$config_file"
    exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        exit 0
    fi
    attempt=$((attempt + 1))
    printf 'supervisor exit=%s restart=%s/3\n' "$exit_code" "$attempt"
    sleep 30
done
exit "$exit_code"
