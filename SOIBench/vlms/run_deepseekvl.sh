#!/usr/bin/env bash
set -e

# =========================
# DeepSeek-VL2 API 推理脚本 (硅基流动)
# =========================

# 设置 API Key (请替换为您的实际 API Key)
export SILICONFLOW_API_KEY='your-siliconflow-api-key-here'

PY_SCRIPT="SOIBench/vlms/run_grounding_deepseekvl.py"

OUTPUT_ROOT="/home/member/data2/wyp/SOT/VLMTrack/SOIBench/results"
EXP_TAG="deepseekvl2-siliconflow"

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_api_${EXP_TAG}.log"

# =========================
# 后台运行
# =========================
nohup python "${PY_SCRIPT}" \
  --mode api \
  --api_model_name deepseek-ai/deepseek-vl2 \
  --api_base_url https://api.siliconflow.cn/v1 \
  --output_root "${OUTPUT_ROOT}" \
  --exp_tag "${EXP_TAG}" \
  --api_temperature 0.1 \
  --api_max_tokens 512 \
  --api_retries 3 \
  > "${LOG_FILE}" 2>&1 &

echo "🚀 DeepSeek-VL2 API 推理已后台启动"
echo "📄 日志文件: ${LOG_FILE}"
echo "🔍 实时查看: tail -f ${LOG_FILE}"

# 停止运行: pkill -f run_grounding_deepseekvl.py
