# AnyGameRenderer: A Video-to-Video Generative Renderer for Games

Coding agents can now turn a one-paragraph prompt into a playable game in minutes — game logic, physics, camera, win
conditions, working HUD. What they cannot produce is good art: the output is gray boxes, colored cylinders and cones on
flat-shaded ground, the "programmer art" that human studios spend months replacing with modeling, texturing and lighting.

**AnyGameRenderer** is the rendering layer for that gap. It takes a **crude gameplay recording** and a **text prompt**,
and outputs the same gameplay re-rendered in the prompted style, frame-aligned with the input. It adds characters,
motion detail and effects, while keeping the layout, the camera, and every entity's position and timing. One
checkpoint, no per-object labels, no engine hooks: anything that can be screen-recorded can be rendered.

Built as a LoRA on [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) Ref2VA (reference-video-to-video+audio,
1344×768, 24 fps). This repository is the inference package: `infer.py`, the caption builder, example clips and a
self-contained copy of the model code.

## Highlights

- 🎮 **Generalizes across genres.** Prior and concurrent renderers are demonstrated mainly on third-person, first-person
  or driving footage. AnyGameRenderer covers the genres agents actually produce — tower defense, auto-battlers, top-down
  survivors, turn-based battles, kart racing, crowd runners, arena survival, co-op games and shooters — with a single
  checkpoint and no per-game tuning.
- 🎯 **Keeps entity correspondence.** The hard part of game rendering is bookkeeping: twelve enemies become nine when
  three die, a reinforcement wave walks in at t = 4.2 s, a tiny far-away mob must not dissolve into fog. Group counts
  stay correct, entities appear and disappear on the right frame, and fewer small entities are lost — where the
  zero-shot base leaves blocks unrendered, drops entities or hallucinates characters that are not in the input.
- 📹 **Its input is plain video.** No per-object class labels, state-augmented 3D boxes or engine proxies. It accepts
  whatever a crude game looks like — gray blockouts, colored primitives (boxes, wireframes, cylinders, ellipsoids),
  semantic-colored maps and coarse clay mockups — at different granularities, with one checkpoint.
- 🎨 **Any prompted style.** The same recording renders as photoreal AAA, LEGO, Ghibli, crayon, synthwave, samurai,
  candy land … and the characters are re-imagined to match, while staying frame-aligned with the input.
- 📊 **Measured at the entity level.** On a held-out benchmark of 59 cases / 753 ground-truth-boxed entities, four
  metrics score whether each entity appears on time, stays in place, and is neither unrendered nor hallucinated.
  Over the zero-shot base: temporal match 50.9 → 62.9 %, position match 57.1 → 61.9 %, unrendered rate 4.1 → 2.9 %,
  hallucination rate 52.1 → 46.0 %.

## Gallery

Crude input on the **left** (top-left in the style grids) · AnyGameRenderer on the **right**, playing in sync.
All clips are 5.2 s at 1344×768; the GIFs are downscaled. One renderer, every genre, input form and art style below.

| | |
|:--:|:--:|
| ![](assets/gallery/genre_trenchfire.gif) | ![](assets/gallery/genre_cityroam.gif) |
| WW1 shooter · first person | Open-city driving · third person |
| ![](assets/gallery/genre_turbokart.gif) | ![](assets/gallery/genre_elitestriker.gif) |
| Kart racing · chase cam | Soccer broadcast · TV cam |
| ![](assets/gallery/genre_spinarena.gif) | ![](assets/gallery/genre_aliensurv.gif) |
| Spinning-top battle · arena | Survivors-like · top down |
| ![](assets/gallery/genre_wheelfactor.gif) | ![](assets/gallery/genre_fallenangels.gif) |
| Vehicle combat · mud arena | Dark fantasy · top down |
| ![](assets/gallery/genre_yavball.gif) | ![](assets/gallery/input_wireframe.gif) |
| Fantasy ball sport · arena | Wireframe-box input |
| ![](assets/gallery/input_coarse.gif) | ![](assets/gallery/input_semantic.gif) |
| Coarse 3D mockup input | 2D semantic map input |
| ![](assets/gallery/input_ellipsoids.gif) | |
| Ellipsoid input | |
| ![](assets/gallery/styles_blockomon.gif) | ![](assets/gallery/styles_turbokart.gif) |
| One blockout, three styles · steampunk / ukiyo-e / felt &amp; knit | One blockout, three styles · synthwave / nuclear wasteland / pirate seas |

## Quick Start

```bash
git clone https://github.com/MickeyLLG/AnyGameRenderer_infer.git
cd AnyGameRenderer_infer

bash setup_env.sh && conda activate agr   # conda env "agr": Python 3.10, torch 2.9.1+cu128, transformers 4.57.6, av …
bash download_weights.sh                  # MiniMax-H3 base weights (~123 GB) + Qwen3-VL processor files -> weights/
cp /path/to/anygamerenderer_v10ep3_lora.safetensors weights/   # the LoRA is distributed separately (2.4 GB)

bash run_example.sh                       # renders examples/*/primitive.mp4 -> examples/out/*.mp4 on GPU 0 (~17 min per clip)
```

Accept the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before
downloading the base weights; `weights/README.md` lists the expected files.

Each `examples/<game>/` holds `primitive.mp4` (the crude recording — colored primitives over a gray blockout), `spec.json`, the `caption.txt` built from it,
`ground_truth.mp4` (the real game footage the crude recording was captured from — never used by the model) and, for two
of them, `rendered_v10ep3.mp4`, the output of this package with the default settings (seed 42).

**Your own clip** is three steps:

```bash
python make_caption.py my_clip/spec.json -o my_clip/caption.txt          # 1. spec -> caption in the training format
echo '{"name":"my_clip","primitive":"my_clip/primitive.mp4","caption_file":"my_clip/caption.txt","out":"out/my_clip.mp4"}' > jobs.jsonl
CUDA_VISIBLE_DEVICES=0 python infer.py jobs.jsonl                        # 2-3. one process per GPU; finished outputs are skipped on re-run
```

`infer.py` options: `--steps 20` `--seed 42` `--blocks_to_swap 40` (lower on GPUs > 100 GB, 0 = no swap) `--lora ''`
(base model only) `--lora_multiplier 1.0` `--attn_mode flash3` (if FlashAttention 3 is installed) `--weights DIR`
`--processor DIR|HF-id`. Output: `out/<name>.mp4`, 1344×768, 24 fps, with the audio track H3 generates for the
soundscape line.

## Inputs and captions

**Crude recording (the input video)** — mp4, 24 fps, 2–15 s; 1344×768 recommended (other 16:9 sizes are resized to the 768-short-edge
canvas). The generated length is the clip length snapped down to `17n+5` frames (124 = 5.17 s, 345 = 15 s max).

- Environment: untextured **pale grey** geometry, no textures or lighting cues beyond simple shading.
- Each character / weapon / vehicle class is a **flat-coloured proxy** (box, wireframe box, cylinder or ellipsoid; one
  shape per clip) moving along the true path of that entity. One distinct, saturated colour per class: red, orange,
  yellow, yellow-green, green, cyan, blue, purple, magenta.
- Nothing else: no HUD, no text, no effects (effects are described in the caption, not drawn).

**Caption** — write a short `spec.json`; `make_caption.py` expands it into the six-section layout the model was trained
on (`python make_caption.py --template` prints a full example; the docstring documents every field):

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

What matters for sync: describe *what each proxy is* and *the setting* concretely; set `"many": true` for any class
with several proxies (crowds, enemies, rival cars); use the optional `"style"` for the art direction (lighting, medium,
era — the gallery looks were produced this way) but never describe geometry that is not in the input recording; write
positive statements only — the model has no negative guidance, so "no HUD" tends to produce a HUD.

## Requirements

| | |
|---|---|
| GPU | 1× ≥ 80 GB (A100 / H100 / H20 class). Peak ≈ 50 GB while the text encoder is loaded, ≈ 32 GB during sampling with `--blocks_to_swap 40`. |
| CPU RAM | ≥ 128 GB: the 66 GB transformer is loaded on CPU for the LoRA merge and swapped blocks stay there. |
| Disk | 123 GB base weights + 2.4 GB LoRA. |
| Time | ≈ 17 min per 124-frame clip at 20 steps on H100-class (SDPA), plus ≈ 6 min one-off weight loading per process. |
| Software | Linux, CUDA 12.8 driver, Python 3.10; pinned packages in `requirements.txt`. |

## Repository layout

```
infer.py            jobs.jsonl -> mp4s; loads the weights once and loops over the jobs
make_caption.py     spec.json -> caption.txt in the exact training format
h3_media.py         white-model video decoding (24 fps resample, 17n+5 frames, model canvas)
third_party/        25-file subset of musubi-tuner (MiniMax-H3 model, VAEs, text encoder, LoRA merge, block swap) — see NOTICE.md
weights/            base weights (download_weights.sh) + the LoRA (distributed separately)
examples/           three white-model clips with spec.json, caption.txt and the original game footage
assets/gallery/     the GIFs above
```

## Notes

- Resolution and length are fixed to what the LoRA was trained on (1344×768, 124 frames); longer clips (up to 345
  frames) run but were not trained for.
- Generation is deterministic for a given seed; different seeds change appearance details, not layout.
- `third_party/musubi_tuner` is taken from [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) at commit
  `c4dc6b8` (its MiniMax-H3 branch). `infer.py` always imports this copy; an installed musubi-tuner is neither needed
  nor used.

## Acknowledgements

Built on [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (Ref2VA backbone and VAEs, weights repacked by
[Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3)) and on [musubi-tuner](https://github.com/kohya-ss/musubi-tuner)
for the MiniMax-H3 model code and LoRA training.

## License

Code in this repository: see `LICENSE` (the vendored `third_party/musubi_tuner` is Apache-2.0, see
`third_party/NOTICE.md`). The base weights are released under the
[MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE); the AnyGameRenderer LoRA is a
derivative of those weights and is distributed under the same terms.

## Citation

```bibtex
@misc{anygamerenderer2026,
  title  = {AnyGameRenderer: Rendering Game Videos from White-Model Blockouts},
  author = {TODO},
  year   = {2026},
  note   = {TODO: paper / project page}
}
```
