#!/usr/bin/env bash
set -euo pipefail

# ===== Parameters =====
VLLM_ENV_PYTHON="${VLLM_ENV_PYTHON:-/workspace/envs/vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/user-data/MODEL_WEIGHTS_PUBLIC/MLLM_weights/Qwen3_5-9B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
API_KEY="${API_KEY:-local-test-key}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":2}}"
LOG_FILE="${LOG_FILE:-/workspace/tmp/vllm_qwen35_9b.log}"
PID_FILE="${PID_FILE:-/workspace/tmp/vllm_qwen35_9b.pid}"

mkdir -p "$(dirname "$LOG_FILE")"

unset PYTHONPATH PIP_TARGET PYTHONUSERBASE
export CUDA_HOME
export VLLM_TARGET_DEVICE=cuda
export PYTHONUNBUFFERED=1

nohup "$VLLM_ENV_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
  >"$LOG_FILE" 2>&1 &

echo $! >"$PID_FILE"
echo "started pid=$(cat "$PID_FILE")"
echo "log=$LOG_FILE"
