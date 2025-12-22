#!/usr/bin/env bash
set -e

# =========================
# 基本配置
# =========================
# export DASHSCOPE_API_KEY='sk-61547e720ce8407aa44f4511051903b0'
# 或者提前在 ~/.bashrc 里 export，这里就可以删掉这一行

PY_SCRIPT="SOIBench/vlms/run_grounding_qwen3vl.py"

OUTPUT_ROOT="/home/member/data2/wyp/SOT/VLMTrack/SOIBench/results"
EXP_TAG="qwen3-vl-32b-instruct"

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_api_${EXP_TAG}.log"

# =========================
# 挂起运行
# =========================
nohup python "${PY_SCRIPT}" \
  --mode api \
  --api_model_name qwen3-vl-32b-instruct \
  --api_base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --output_root "${OUTPUT_ROOT}" \
  --exp_tag "${EXP_TAG}" \
  --save_debug_vis \
  --api_temperature 0.1 \
  --api_max_tokens 512 \
  --api_retries 3 \
  > "${LOG_FILE}" 2>&1 &

echo "🚀 API 推理已后台启动"
echo "📄 日志文件: ${LOG_FILE}"
echo "🔍 实时查看: tail -f ${LOG_FILE}"


# pkill -f run_grounding_qwen3vl.py
# nohup bash run_api_vlm_grounding.sh > /dev/null 2>&1 &
