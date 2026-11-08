#!/usr/bin/env bash
# 【非核心】统一启动项目只读看板，拒绝使用原Demo的第二套计算入口。
set -euo pipefail
board_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(cd "$board_root/../.." && pwd)"
exec "$project_root/.venv/bin/python" "$board_root/scripts/project_preview.py" serve "$@"
