#!/usr/bin/env bash
set -euo pipefail

cd /root/user-data/wyp/VLMTrack

export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-sk-61547e720ce8407aa44f4511051903b0}"

/workspace/envs/wyp_sot/bin/python tracking/test.py \
  vlm_cognitive_mosaic \
  api_qwen3-vl-235b-a22b-instruct_v2 \
  --dataset_name videocube_val_tiny \
  --threads 4 \
  --debug 2 \
  --num_gpus 1 \
  --run_tag videocube_val_tiny__api_qwen3-vl-235b-a22b-instruct_v2__threads4
