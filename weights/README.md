# weights/

Expected layout (run `../download_weights.sh` for the base weights; the LoRA is distributed separately, not in git):

```
weights/
  anygamerenderer_v10ep3_lora.safetensors        2.4 GB  AnyGameRenderer LoRA (distributed separately, not in git)
  diffusion_models/minimax_h3_ref2va_bf16.safetensors      66 GB  MiniMax-H3 Ref2VA transformer, BF16
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors    52 GB  Qwen3-VL-32B text encoder as released with H3
  vae/minimax_h3_video_vae_fp16.safetensors                5.2 GB
  vae/minimax_h3_audio_vae_fp32.safetensors                0.6 GB
  processors/Qwen3-VL-32B-Instruct/                        11 MB  tokenizer + processor configs (no model weights)
```

Base weights: https://huggingface.co/Comfy-Org/MiniMax-H3 (MiniMax-H3 Community License).
Processor: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct.

## LoRA

`anygamerenderer_v10ep3_lora.safetensors` — rank 128, alpha 128, on `attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2`
of every transformer block (musubi-tuner `networks.lora_minimax_h3` format, 600 tensors). Trained on the Ref2VA base
model with the standard flow-matching loss; epoch 3 of the v10 run (gear-pair + UE + pwm-pair mixture, 1344x768, 124 frames).
`infer.py` merges it into the BF16 transformer at load time (`--lora_multiplier 1.0`).
