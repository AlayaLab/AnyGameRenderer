"""AnyGameRenderer inference: white-model (blockout) video + caption -> rendered game video.

Runs MiniMax-H3 ref2va with the AnyGameRenderer LoRA merged in. Weights are loaded once; every job in the
JSONL is then generated in turn (text encoder first for all jobs, then transformer + VAEs stay resident).

Usage:
    python infer.py jobs.jsonl [--weights weights] [--lora weights/anygamerenderer_lora.safetensors]
                               [--steps 20] [--seed 42] [--blocks_to_swap 40] [--width 1344 --height 768]

jobs.jsonl, one JSON object per line:
    {"name": "clip_a", "primitive": "clip_a/primitive.mp4", "caption_file": "clip_a/caption.txt", "out": "out/clip_a.mp4"}
    name           any label (used in logs)
    primitive      the white-model video, 24 fps, 2-15 s (see README for how to make one)
    caption        the caption text, or caption_file: path to a text file holding it (see make_caption.py)
    out            output mp4 (video + generated audio track); existing outputs are skipped
    frames         optional, number of frames to generate; default = whole primitive, snapped to 17n+5 (max 345)
Relative paths in a job are resolved from the directory of the JSONL file.
"""
import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "third_party"))  # vendored musubi_tuner (see third_party/NOTICE.md)

import torch
from safetensors.torch import load_file
from musubi_tuner.minimax_h3.audio_vae import load_audio_vae
from musubi_tuner.minimax_h3.media import H3AudioSource, H3Record, H3Reference, audio_latent_frames, video_latent_frames
from musubi_tuner.minimax_h3.model import load_h3_transformer
from musubi_tuner.minimax_h3.packing import H3ReferenceGeometry, H3VideoGeometry, build_h3_layout
from musubi_tuner.minimax_h3.sampling import augment_condition_latents, initialize_target_latents, sample_joint_av, synchronize_decoded_av, write_joint_av
from musubi_tuner.minimax_h3.text_encoder import H3TextVisual, build_presentation, encode_h3_presentation, load_h3_processor, load_h3_text_encoder
from musubi_tuner.minimax_h3.video_vae import encode_video_condition, load_video_vae
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.networks import lora_minimax_h3
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.lora_utils import filter_lora_state_dict

from h3_media import decode_reference_video, video_duration_seconds

MAX_FRAMES = 345                 # 15 s at 24 fps, the released upper limit
VAE_SPATIAL_RATIO = 16           # pixels per video-latent cell
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 3.0           # MiniMax-H3 flow-schedule shifts
VISUAL_COND_CLEAN, AUDIO_COND_CLEAN = 0.999, 1.0


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def snap_frames(n):
    """MiniMax-H3 frame counts must be 17*n+5."""
    return min(MAX_FRAMES, 5 + ((n - 5) // 17) * 17)


def read_jobs(jsonl):
    base = Path(jsonl).resolve().parent
    jobs = []
    for line in open(jsonl):
        if not line.strip():
            continue
        j = json.loads(line)
        prim = (base / j["primitive"]).resolve()
        if not prim.is_file():
            raise FileNotFoundError(f"{j['name']}: primitive video not found: {prim}")
        caption = j["caption"] if "caption" in j else open(base / j["caption_file"]).read().strip()
        duration = video_duration_seconds(prim)
        if not 2.0 <= duration <= 15.0:
            raise ValueError(f"{j['name']}: primitive must be 2-15 s long, got {duration:.2f} s")
        frames = int(j.get("frames") or snap_frames(int(duration * 24 + 1e-6)))
        video_latent_frames(frames)  # raises unless 17n+5
        jobs.append(dict(name=j["name"], primitive=prim, caption=caption, frames=frames, duration=duration, out=(base / j["out"]).resolve()))
    return jobs


def make_record(job):
    """A ref2va record whose only reference is the white-model video. No target video or audio is needed."""
    ref = H3Reference(type="video", path=job["primitive"], audio=None, duration_seconds=job["duration"])
    return H3Record(video_path=job["primitive"], caption=job["caption"], target_audio=H3AudioSource(job["primitive"], embedded=False), references=(ref,), jsonl_line=0)


def merge_lora(transformer, path, multiplier):
    state = filter_lora_state_dict(load_file(path), None, None)
    network = lora_minimax_h3.create_arch_network_from_weights(multiplier, state, unet=transformer, for_inference=True)
    if not network.unet_loras:
        raise ValueError(f"LoRA {path} has no modules matching this transformer")
    network.merge_to(None, transformer, state, dtype=torch.bfloat16, device="cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs")
    ap.add_argument("--weights", default=str(HERE / "weights"), help="directory with diffusion_models/ text_encoders/ vae/ processors/ (see download_weights.sh)")
    ap.add_argument("--lora", default=str(HERE / "weights/anygamerenderer_lora.safetensors"), help="AnyGameRenderer LoRA; pass '' for the base model only")
    ap.add_argument("--lora_multiplier", type=float, default=1.0)
    ap.add_argument("--processor", default=None, help="Qwen3-VL processor dir or HF id (default: <weights>/processors/Qwen3-VL-32B-Instruct, else Qwen/Qwen3-VL-32B-Instruct)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blocks_to_swap", type=int, default=40, help="transformer blocks kept in CPU RAM during sampling; 40 fits an 80 GB GPU, 0 needs ~100 GB")
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--attn_mode", default="torch", choices=("torch", "flash3"), help="flash3 needs flash-attn 3 installed")
    cli = ap.parse_args()
    if cli.width % 32 or cli.height % 32:
        sys.exit("width and height must be multiples of 32")

    W = Path(cli.weights)
    paths = dict(dit=W / "diffusion_models/minimax_h3_ref2va_bf16.safetensors", text_encoder=W / "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
                 video_vae=W / "vae/minimax_h3_video_vae_fp16.safetensors", audio_vae=W / "vae/minimax_h3_audio_vae_fp32.safetensors")
    for k, p in paths.items():
        if not p.is_file():
            sys.exit(f"missing {k} weights: {p}  (run download_weights.sh)")
    if cli.lora and not Path(cli.lora).is_file():
        sys.exit(f"missing LoRA: {cli.lora}")
    processor = cli.processor or (str(W / "processors/Qwen3-VL-32B-Instruct") if (W / "processors/Qwen3-VL-32B-Instruct").is_dir() else "Qwen/Qwen3-VL-32B-Instruct")

    jobs = [j for j in read_jobs(cli.jobs) if not j["out"].exists()]
    if not jobs:
        log("nothing to do (all outputs exist)")
        return
    device = torch.device("cuda")
    log(f"{len(jobs)} job(s), lora={cli.lora or 'none'}, {cli.width}x{cli.height}, steps={cli.steps}, seed={cli.seed}")

    # Phase 1: text conditioning for every job (Qwen3-VL text encoder loaded once, then freed).
    t0 = time.time()
    proc = load_h3_processor(processor, revision=None)
    text_encoder = load_h3_text_encoder(str(paths["text_encoder"]), processor_path=processor, revision=None, device=device, dtype=torch.bfloat16, disable_mmap=False)
    log(f"text encoder loaded in {time.time() - t0:.0f}s")
    text_cond = {}
    for j in jobs:
        try:
            record = make_record(j)
            frames = decode_reference_video(j["primitive"], j["frames"])
            sampled = frames[::12]  # the text encoder sees one frame every 0.5 s
            visuals = {record.references[0].path: H3TextVisual(sampled, tuple(i / 2.0 for i in range(sampled.shape[0])))}
            hs, tags = encode_h3_presentation(proc, text_encoder, build_presentation(record, "ref2va", visuals))
            text_cond[j["name"]] = (hs.to(torch.bfloat16).unsqueeze(0).cpu(), tags.cpu())
            log(f"text ok {j['name']} ({hs.shape[0]} tokens, {j['frames']} frames)")
        except Exception:
            log(f"text FAILED {j['name']}")
            traceback.print_exc()
    del proc, text_encoder
    gc.collect()
    clean_memory_on_device(device)

    # Phase 2: transformer (+ merged LoRA, block swap) and both VAEs stay resident; generate job by job.
    t0 = time.time()
    transformer = load_h3_transformer(str(paths["dit"]), device="cpu", dtype=torch.bfloat16, attn_mode=cli.attn_mode, split_attn=False, disable_mmap=False)
    if cli.lora:
        merge_lora(transformer, cli.lora, cli.lora_multiplier)
    if cli.blocks_to_swap:
        transformer.enable_block_swap(cli.blocks_to_swap, BlockSwapConfig(device=device, supports_backward=False, use_pinned_memory=False))
        transformer.move_to_device_except_swap_blocks(device)
        transformer.prepare_block_swap_before_forward()
        transformer.switch_block_swap_for_inference()
    else:
        transformer.to(device)
    transformer.eval().requires_grad_(False)
    video_vae = load_video_vae(str(paths["video_vae"]), device=device, dtype=torch.float16, disable_mmap=False)
    audio_vae = load_audio_vae(str(paths["audio_vae"]), device=device, dtype=torch.float32, disable_mmap=False)
    log(f"transformer + VAEs loaded in {time.time() - t0:.0f}s; GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB")

    n_ok = 0
    for j in jobs:
        if j["name"] not in text_cond or j["out"].exists():
            continue
        t1 = time.time()
        try:
            record = make_record(j)
            # Visual condition: the white-model video through the video VAE.
            pixels = decode_reference_video(j["primitive"], j["frames"]).float().div(127.5).sub(1.0).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
            with torch.no_grad():
                cond = encode_video_condition(video_vae, pixels.to(device, torch.float16)).cpu()
            del pixels
            cond_geometry = H3VideoGeometry(*cond.shape[2:])
            hs, tags = text_cond[j["name"]]
            layout = build_h3_layout(task="ref2va", text_length=hs.shape[1],
                                     target_video=H3VideoGeometry(video_latent_frames(j["frames"]), cli.height // VAE_SPATIAL_RATIO, cli.width // VAE_SPATIAL_RATIO),
                                     target_audio_frames=audio_latent_frames(j["frames"]), visual_conditions=(),
                                     references=[H3ReferenceGeometry("video", video=cond_geometry, audio_frames=0)])
            init_v, init_a = initialize_target_latents(video_shape=(1, 24, layout.target_video.frames, layout.target_video.height, layout.target_video.width),
                                                       audio_shape=(1, 32, 2, layout.target_audio_frames), seed=cli.seed, device=device, video_dtype=torch.float32, audio_dtype=torch.float32)
            visual_conditions, audio_conditions = augment_condition_latents((cond,), (), seed=cli.seed, visual_clean=VISUAL_COND_CLEAN, audio_clean=AUDIO_COND_CLEAN, device=device)
            if transformer.offloader is not None:
                transformer.prepare_block_swap_before_forward()
            sample = sample_joint_av(transformer, layout=layout, text_hidden_states=hs.to(device=device, dtype=torch.bfloat16), text_token_tags=tags.unsqueeze(0).to(device),
                                     initial_video=init_v, initial_audio=init_a, steps=cli.steps, video_shift=SHIFT_VIDEO, audio_shift=SHIFT_AUDIO,
                                     visual_condition_latents=visual_conditions, audio_condition_latents=audio_conditions,
                                     visual_condition_clean=VISUAL_COND_CLEAN, audio_condition_clean=AUDIO_COND_CLEAN,
                                     step_callback=lambda k, n: log(f"  step {k}/{n} {time.time() - t1:.0f}s") if k in (1, 5, 10, 15, n) else None)
            del cond, visual_conditions, audio_conditions, init_v, init_a
            clean_memory_on_device(device)
            with torch.no_grad():
                dv = video_vae.decode(sample.video.to(device, torch.float16)).cpu()
                da = audio_vae.decode(sample.audio.to(device)).cpu()
            del sample
            clean_memory_on_device(device)
            j["out"].parent.mkdir(parents=True, exist_ok=True)
            write_joint_av(synchronize_decoded_av(dv, da, frame_count=j["frames"]), str(j["out"]))
            del dv, da
            gc.collect()
            clean_memory_on_device(device)
            n_ok += 1
            log(f"ok {j['name']} {time.time() - t1:.0f}s -> {j['out']}  (GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)")
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            log(f"FAILED {j['name']} after {time.time() - t1:.0f}s")
            traceback.print_exc()
            gc.collect()
            clean_memory_on_device(device)
    log(f"done: {n_ok}/{len(jobs)} ok")


if __name__ == "__main__":
    main()
