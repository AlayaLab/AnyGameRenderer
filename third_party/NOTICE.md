# third_party/musubi_tuner

A trimmed copy of [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) — only the 25 files that
MiniMax-H3 ref2va inference needs (`minimax_h3/`, `networks/lora*.py`, `modules/`, `utils/`, `wan/modules/attention.py`).
Source commit: `c4dc6b8038c114f730f1aa24a98064de3bd62437` ("fix: simplify MiniMax-H3 R1 batching and provenance", 2026-08-03,
MiniMax-H3 support branch; not on `main` at the time of writing).

Local changes relative to that commit (all in `minimax_h3/`): `model.py`, `packing.py`, `sampling.py` gain an optional
per-row denoise mask ("retake") path. It is not used by `infer.py`; with the mask left unset the code path is identical to upstream.

License: Apache License 2.0 (musubi-tuner; the MiniMax-H3 files are adapted from Diffusers PR #14355 and keep their headers).
`wan/modules/attention.py` is modified from Wan2.1, Apache License 2.0.
