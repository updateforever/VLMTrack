#!/bin/bash
# 认知跟踪测试脚本

# 设置环境
export PYTHONPATH=/data/wyp/VLMTrack:$PYTHONPATH
export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}

# 测试数据集
DATASET="videocube_val"  # MGIT val 集
TRACKER="qwen_vlm_cognitive"
CONFIG="default"  # 使用 API 模式

echo "========================================="
echo "认知跟踪测试"
echo "========================================="
echo "Tracker: $TRACKER"
echo "Config: $CONFIG"
echo "Dataset: $DATASET"
echo "========================================="

# 运行测试（单序列调试）
python tracking/test.py $TRACKER $CONFIG \
    --dataset $DATASET \
    --debug 1 \
    --threads 1

echo ""
echo "========================================="
echo "测试完成"
echo "========================================="
echo "结果保存在: ./results/$TRACKER/$DATASET/"
echo ""
echo "查看跟踪历史（如果保存了）："
echo "  - target_status 分布"
echo "  - environment_status 分布"
echo "  - tracking_evidence 示例"
