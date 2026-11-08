#!/usr/bin/env bash
set -euo pipefail
project=/root/shared-nvme/hft_mm_research
mkdir -p "$project/workspace/results/factorboard/service"
exec /base/mambaforge/bin/supervisord -c "$project/apps/factorboard/deploy/supervisord.conf"
