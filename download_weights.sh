#!/usr/bin/env bash
# Download into weights/: the MiniMax-H3 base weights (~123 GB), the Qwen3-VL processor files (~11 MB), and the
# AnyGameRenderer LoRA (2.4 GB, from AlayaLab/AnyGameRenderer).
#
# Read and accept the MiniMax-H3 Community License first:
#   https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
# Run inside the "agr" env from setup_env.sh (huggingface_hub 0.34.3, `huggingface-cli`); newer huggingface_hub (>= 1.0,
# `hf` instead of `huggingface-cli`) also works — the script picks whichever CLI exists. Set HF_TOKEN if a repo asks for login.
set -euo pipefail
W="$(cd "$(dirname "$0")" && pwd)/weights"
mkdir -p "$W"
if command -v huggingface-cli >/dev/null 2>&1; then HF="huggingface-cli"; elif command -v hf >/dev/null 2>&1; then HF="hf"; else
  echo "need huggingface-cli or hf: pip install 'huggingface_hub[cli]==0.34.3' (or activate the agr env)"; exit 1; fi

# Base model, Comfy-Org repack of the released BF16 checkpoints (same file names as in weights/README.md).
$HF download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir "$W"

# Tokenizer / processor configs only (the 32B text-encoder weights above are the H3 release, not Qwen's own).
# One --include per call: the CLI's handling of several patterns differs between huggingface_hub versions.
P="$W/processors/Qwen3-VL-32B-Instruct"
$HF download Qwen/Qwen3-VL-32B-Instruct --include "*.json" --local-dir "$P"
$HF download Qwen/Qwen3-VL-32B-Instruct --include "merges.txt" --local-dir "$P"
for f in config.json tokenizer.json tokenizer_config.json preprocessor_config.json video_preprocessor_config.json merges.txt; do
  [ -s "$P/$f" ] || { echo "processor file missing: $P/$f"; exit 1; }
done

# AnyGameRenderer LoRA.
$HF download AlayaLab/AnyGameRenderer anygamerenderer_lora.safetensors --local-dir "$W"

echo "weights ready:"; du -sh "$W"/*
