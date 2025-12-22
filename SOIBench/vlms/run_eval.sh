#!/bin/bash
# SOIBench Grounding 评测示例脚本

# ============================================
# 1. 运行 Grounding 推理
# ============================================
echo "=========================================="
echo "步骤 1: 运行 Grounding 推理"
echo "=========================================="

# 使用本地模型
python run_grounding_qwen3vl.py \
    --mode local \
    --model_path /path/to/Qwen2-VL-7B-Instruct \
    --lasot_jsonl /path/to/lasot_jsonl \
    --lasot_root /path/to/lasot_images \
    --mgit_jsonl /path/to/mgit_jsonl \
    --mgit_root /path/to/mgit_images \
    --tnl2k_jsonl /path/to/tnl2k_jsonl \
    --tnl2k_root /path/to/tnl2k_images \
    --output_root ./results \
    --exp_tag v1 \
    --save_debug_vis

# 或使用 API
# python run_grounding_qwen3vl.py \
#     --mode api \
#     --api_model_name qwen-vl-max \
#     --api_key_env YOUR_API_KEY \
#     --lasot_jsonl /path/to/lasot_jsonl \
#     --lasot_root /path/to/lasot_images \
#     --output_root ./results \
#     --exp_tag api_v1

# ============================================
# 2. 评测结果
# ============================================
echo ""
echo "=========================================="
echo "步骤 2: 评测结果"
echo "=========================================="

python eval_results.py \
    --pred_root ./results \
    --output_dir ./eval_results \
    --models local_v1 \
    --datasets lasot mgit tnl2k \
    --lasot_gt_root /path/to/lasot_jsonl \
    --mgit_gt_root /path/to/mgit_jsonl \
    --tnl2k_gt_root /path/to/tnl2k_jsonl

# ============================================
# 3. 可视化结果 (可选)
# ============================================
echo ""
echo "=========================================="
echo "步骤 3: 可视化结果 (可选)"
echo "=========================================="

# 可视化单个序列为图片
python visualize_grounding.py \
    --dataset lasot \
    --seq_name airplane-1 \
    --pred_file ./results/lasot/local_v1/airplane-1_pred.jsonl \
    --gt_file /path/to/lasot_jsonl/airplane-1_descriptions.jsonl \
    --image_root /path/to/lasot_images \
    --output_dir ./vis_results

# 或保存为视频
# python visualize_grounding.py \
#     --dataset lasot \
#     --seq_name airplane-1 \
#     --pred_file ./results/lasot/local_v1/airplane-1_pred.jsonl \
#     --gt_file /path/to/lasot_jsonl/airplane-1_descriptions.jsonl \
#     --image_root /path/to/lasot_images \
#     --output_dir ./vis_results \
#     --save_video \
#     --fps 30

echo ""
echo "✅ 全部完成!"
