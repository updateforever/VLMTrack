#!/usr/bin/env bash
set -euo pipefail

cd /root/user-data/wyp/VLMTrack

export LOCAL_VLLM_BASE_URL=http://127.0.0.1:8002/v1
export LOCAL_VLLM_API_KEY=local-test-key

/workspace/envs/wyp_sot/bin/python tracking/test.py \
  vlm_cognitive_mosaic \
  api_vllm_qwen25_vl_7b_v2 \
  --dataset_name videocube_val_tiny \
  --threads 0 \
  --debug 1 \
  --num_gpus 1 \
  --run_tag videocube_val_tiny__api_vllm_qwen25_vl_7b_v2
