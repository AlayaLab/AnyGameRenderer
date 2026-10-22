#!/usr/bin/env bash
# Create the inference environment (conda, Python 3.10, CUDA 12.8 wheels).
#   bash setup_env.sh            -> conda env "agr"
#   ENV=myname bash setup_env.sh
set -euo pipefail
ENV=${ENV:-agr}
HERE="$(cd "$(dirname "$0")" && pwd)"

conda create -y -n "$ENV" python=3.10
PY="$(conda run -n "$ENV" which python)"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r "$HERE/requirements.txt"
# Optional, ~30% faster sampling on Hopper GPUs: FlashAttention 3, then run infer.py with --attn_mode flash3.
# Prebuilt wheels: https://github.com/mjun0812/flash-attention-prebuild-wheels  (needs the flash_attn_3 / flash_attn_interface module)

"$PY" - <<'EOF'
import torch, transformers, av
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "| av", av.__version__)
EOF
echo "env '$ENV' ready: conda activate $ENV"
