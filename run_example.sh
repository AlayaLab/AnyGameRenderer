#!/usr/bin/env bash
# Render the three bundled examples: build captions from their spec.json, then run infer.py on one GPU.
#   bash run_example.sh            # GPU 0
#   GPU=2 bash run_example.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
for d in examples/*/; do
  python make_caption.py "$d/spec.json" -o "$d/caption.txt"
done
CUDA_VISIBLE_DEVICES=${GPU:-0} python infer.py examples/jobs.jsonl
ls -la examples/out/
