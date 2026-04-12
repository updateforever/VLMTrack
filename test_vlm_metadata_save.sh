#!/bin/bash

# 测试 VLM 元数据保存功能
# 在一个小序列上运行，验证 JSON 文件是否正确生成

echo "Testing VLM metadata save functionality..."
echo "Running vlm_cognitive_mosaic on a short sequence..."

# 使用 debug_frames 限制帧数，加快测试
python tracking/test.py vlm_cognitive_mosaic default \
    --dataset lasot \
    --debug 2 \
    --threads 1 \
    --debug_frames 10

echo ""
echo "Test completed. Checking results..."

# 查找生成的 JSON 文件
RESULTS_DIR="./results/vlm_cognitive_mosaic/default/lasot"

if [ -d "$RESULTS_DIR" ]; then
    echo "Results directory found: $RESULTS_DIR"

    # 查找第一个 _full.json 文件
    JSON_FILE=$(find "$RESULTS_DIR" -name "*_full.json" | head -1)

    if [ -n "$JSON_FILE" ]; then
        echo ""
        echo "✓ Found JSON metadata file: $JSON_FILE"
        echo ""
        echo "Sample content (first 50 lines):"
        head -50 "$JSON_FILE"
        echo ""
        echo "✓ Metadata save test PASSED"
    else
        echo ""
        echo "✗ No _full.json file found in $RESULTS_DIR"
        echo "✗ Metadata save test FAILED"
    fi
else
    echo "✗ Results directory not found: $RESULTS_DIR"
    echo "✗ Metadata save test FAILED"
fi
