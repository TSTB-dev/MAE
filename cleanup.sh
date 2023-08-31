#!/bin/bash

# 引数から対象となるディレクトリを取得
TARGET_DIR="$1"

# 引数が指定されているか確認
if [ -z "$TARGET_DIR" ]; then
    echo "Usage: $0 <target_directory>"
    exit 1
fi

# サブディレクトリを探索
for dir in "$TARGET_DIR"/*; do
    if [ -d "$dir/ckpt" ]; then
        # ckptフォルダが空であるかどうかをチェック
        if [ -z "$(ls -A "$dir/ckpt")" ]; then
            echo "Removing empty directory: $dir"
            rm -r "$dir"
        fi
    fi
done