#!/usr/bin/env bash
set -euo pipefail
# 【非核心】使用现有SSH主机配置访问服务器回环服务，浏览器资料留在原地址。
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:8765:127.0.0.1:8765 qiye-rtx3090
