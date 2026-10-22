#!/usr/bin/env bash
# Download the MiniMax-H3 base weights (~123 GB) and the Qwen3-VL processor files (~11 MB) into weights/.
# The AnyGameRenderer LoRA (weights/anygamerenderer_lora.safetensors, 2.4 GB) is distributed separately; place it in weights/.
#
# Read and accept the MiniMax-H3 Community License first:
#   https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
# Needs `pip install -U "huggingface_hub[cli]"`; set HF_TOKEN if the repos ask for login.
set -euo pipefail
W="$(cd "$(dirname "$0")" && pwd)/weights"
mkdir -p "$W"

# Base model, Comfy-Org repack of the released BF16 checkpoints (same file names as in weights/README.md).
huggingface-cli download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir "$W"

# Tokenizer / processor configs only (the 32B text-encoder weights above are the H3 release, not Qwen's own).
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --include "*.json" "merges.txt" \
  --local-dir "$W/processors/Qwen3-VL-32B-Instruct"

echo "weights ready:"; du -sh "$W"/*
