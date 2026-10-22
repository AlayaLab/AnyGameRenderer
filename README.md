# AnyGameRenderer — inference

Turn a **white-model video** (untextured grey blockout of a game scene with flat-coloured proxy shapes for
characters, weapons and vehicles) plus a **caption** into a rendered game video. The camera, layout and every
proxy's position and timing are preserved; appearance comes from the caption.

Model: [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) Ref2VA (reference-video-to-video+audio, 1344×768, 24 fps)
with the AnyGameRenderer LoRA (`weights/anygamerenderer_v10ep3_lora.safetensors`) merged in.
The white-model video is passed as the single video reference; the caption uses H3's "video editing" format.

```
infer.py            generate: jobs.jsonl -> mp4s (loads weights once, loops over jobs)
make_caption.py     spec.json -> caption.txt in the exact training format
h3_media.py         white-model video decoding (24 fps resample, 17n+5 frames, model canvas)
third_party/        trimmed musubi-tuner (MiniMax-H3 model, VAEs, text encoder, LoRA merge, block swap) — see NOTICE.md
weights/            LoRA (distributed separately) + base weights (download_weights.sh)
examples/           three white-model clips with spec.json, caption.txt and the original game footage (ground_truth.mp4)
```

## 1. Requirements

| | |
|---|---|
| GPU | 1× ≥ 80 GB (A100/H100/H20 class). Peak ≈ 50 GB over a run (text-encoder phase); ≈ 32 GB during sampling with `--blocks_to_swap 40`. |
| CPU RAM | ≥ 128 GB: the 66 GB transformer is loaded on CPU for the LoRA merge and swapped blocks stay there. |
| Disk | 125 GB of base weights + 2.4 GB LoRA. |
| Time | ≈ 17 min per 124-frame (5.2 s) clip at 20 steps on H100-class (SDPA); + ≈ 6 min one-off weight loading. |
| Software | Linux, CUDA 12.8 driver, Python 3.10. |

## 2. Setup

```bash
bash setup_env.sh                 # conda env "agr": torch 2.9.1+cu128, transformers 4.57.6, av, ...  (requirements.txt)
conda activate agr
bash download_weights.sh          # base weights from HF Comfy-Org/MiniMax-H3 + Qwen3-VL processor files -> weights/
```

Accept the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before
downloading. `weights/README.md` lists the expected files and the LoRA details.

## 3. Run the examples

```bash
bash run_example.sh               # GPU=0; builds examples/*/caption.txt from spec.json, renders to examples/out/*.mp4
```

Each `examples/<game>/` holds `primitive.mp4` (the white model), `spec.json`, `caption.txt`, `ground_truth.mp4`
(the real game footage the white model was captured from, for comparison only — it is never used by the model) and,
for two of them, `rendered_v10ep3.mp4` — the output this package produced with the default settings (seed 42).

## 4. Run on your own clips

### 4.1 White-model video

* mp4, 24 fps, 2–15 s; 1344×768 recommended (other 16:9 sizes are resized to the 768-short-edge canvas). The number
  of generated frames is the clip length snapped down to `17n+5` (124 = 5.17 s, 345 = 15 s max).
* Environment: untextured **pale grey** geometry, no textures or lighting cues beyond simple shading.
* Each character / weapon / vehicle class is a **flat-coloured proxy** (box, wireframe box, cylinder or ellipsoid; one
  shape per clip) that moves along the true path of that entity. One distinct colour per class, saturated and
  unambiguous: red, orange, yellow, yellow-green, green, cyan, blue, purple, magenta.
* Nothing else: no HUD, no text, no effects (effects are described in the caption, not drawn).

### 4.2 Caption

Write a short `spec.json` and let `make_caption.py` expand it — the model expects the exact six-section layout it
was trained on, and the script reproduces it verbatim (see the docstring for every field):

```json
{
  "shape": "cylinder",
  "setting": "a futuristic indoor training arena: smooth white curved walls, dark teal panels, red floor stripes ...",
  "subjects": [
    {"color": "green", "desc": "the player's own weapon seen in first person: a futuristic assault rifle ...",
     "many": false, "motion": "locked to the viewpoint, swaying gently with each step"},
    {"color": "blue", "desc": "a rival cadet in a dark-grey combat suit with a visored helmet, carrying a rifle",
     "many": true, "motion": "moving as recorded"}
  ],
  "sound": "hum of arena ventilation, footsteps on hard flooring, rifle fire echoing"
}
```

```bash
python make_caption.py spec.json -o caption.txt      # python make_caption.py --template  prints a full example
```

Guidelines that matter for sync quality: describe *what the proxy is* (`desc`) and *the setting* concretely; set
`"many": true` for any class with several proxies (crowds, enemies, rival cars); keep an optional `"style"` to
lighting / weather / colour grading only — never describe geometry that is not in the white model; write positive
statements only (the base model has no negative guidance: "no HUD" tends to produce a HUD).

### 4.3 Generate

```bash
# jobs.jsonl — one job per line, paths relative to the jsonl file
{"name": "clip_a", "primitive": "clip_a/primitive.mp4", "caption_file": "clip_a/caption.txt", "out": "out/clip_a.mp4"}

CUDA_VISIBLE_DEVICES=0 python infer.py jobs.jsonl
```

Options: `--steps 20` `--seed 42` `--blocks_to_swap 40` (lower it on GPUs > 100 GB for a little speed, 0 = no swap)
`--lora ''` (base model only) `--lora_multiplier 1.0` `--attn_mode flash3` (if FlashAttention 3 is installed)
`--weights DIR` `--processor DIR|HF-id`. Existing outputs are skipped, so a job list can be re-run after a crash.
Run one process per GPU with a different `jobs.jsonl` each to parallelise.

Output: `out/<name>.mp4`, 1344×768, 24 fps, with the audio track H3 generates for the soundscape line.

## 5. Notes

* Resolution and length are fixed to what the LoRA was trained on (1344×768, 124 frames); longer clips (up to 345
  frames) work but were not trained for.
* The model is deterministic for a given seed; different seeds change appearance details, not layout.
* `third_party/musubi_tuner` is a 25-file subset of [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)
  at commit `c4dc6b8` (its MiniMax-H3 branch), Apache-2.0 — `infer.py` always imports this copy, an installed
  musubi-tuner is not needed and is not used.
