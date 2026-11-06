#!/bin/sh
# 【核心逻辑26】下载一个免费样本日，先写入.part文件再发布完整行情文件。
set -eu
project_directory=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
raw_directory="$project_directory/workspace/data/raw"
mkdir -p "$raw_directory"
for data_type in incremental_book_L2 trades; do
    filename="$raw_directory/binance-futures_${data_type}_2025-02-01_BTCUSDT.csv.gz"
    if [ -f "$filename" ]; then
        printf '使用已有文件：%s\n' "$filename"
        continue
    fi
    curl --fail --location --retry 3 --retry-all-errors \
        "https://datasets.tardis.dev/v1/binance-futures/$data_type/2025/02/01/BTCUSDT.csv.gz" \
        --output "$filename.part"
    mv "$filename.part" "$filename"
done
