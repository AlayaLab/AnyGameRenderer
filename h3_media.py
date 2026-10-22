"""Decode a white-model video into the [F,H,W,3] uint8 frames MiniMax-H3 expects as a video reference.

Adapted from musubi-tuner's `minimax_h3_cache_latents.PyAVH3MediaDecoder.decode_reference_visual` so that the
dataset / caching machinery is not needed at inference. Behavior is identical: resample to 24 fps, keep at most
`frame_count` frames snapped down to 17n+5, resize to the model canvas (short edge 768, multiples of 32, <= 768x1344).
"""
from bisect import bisect_left
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image
import torch

TARGET_FPS = 24
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344


def video_duration_seconds(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return float(s.duration * s.time_base) if s.duration is not None else len(decode_frames(path)[0]) / TARGET_FPS


def decode_frames(path):
    """All frames of the video as RGB arrays, with their timestamps and the nominal frame duration."""
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream: {path}")
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else TARGET_FPS
        frame_duration = 1.0 / rate if rate > 0 else 1.0 / TARGET_FPS
        frames, timestamps = [], []
        for index, frame in enumerate(container.decode(stream)):
            timestamps.append(float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base is not None else index * frame_duration)
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"decoded no frames: {path}")
    return frames, timestamps, frame_duration


def resample_indices(timestamps, frame_duration, target_fps=TARGET_FPS):
    """Nearest-frame index for every target_fps tick covering the source duration."""
    origin = timestamps[0]
    t = [x - origin for x in timestamps]
    duration = t[-1] + frame_duration
    count = max(1, math.ceil(duration * target_fps - 1e-9))
    out = []
    for i in range(count):
        tt = i / target_fps
        right = bisect_left(t, tt)
        if right == 0:
            out.append(0)
        elif right == len(t):
            out.append(len(t) - 1)
        else:
            left = right - 1
            out.append(left if tt - t[left] <= t[right] - tt + 1e-12 else right)
    return out


def round_to_multiple(value, multiple=CANVAS_MULTIPLE):
    return max(multiple, round(value / multiple) * multiple)


def adapt_canvas(width, height):
    ratio = width / height
    w, h = (BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE) if ratio >= 1.0 else (BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio)
    if w * h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (w * h))
        w, h = w * s, h * s
    return round_to_multiple(w), round_to_multiple(h)


def decode_reference_video(path, frame_count):
    """[F,H,W,3] uint8 tensor: 24 fps, F = min(available, frame_count) snapped to 17n+5, model canvas size."""
    frames, timestamps, frame_duration = decode_frames(Path(path))
    frames = [frames[i] for i in resample_indices(timestamps, frame_duration)]
    usable = min(len(frames), frame_count)
    if usable < 5:
        raise ValueError(f"reference video needs at least 5 frames: {path}")
    usable = 5 + ((usable - 5) // 17) * 17
    frames = frames[:usable]
    src_h, src_w = frames[0].shape[:2]
    width, height = adapt_canvas(src_w, src_h)
    if src_w * src_h < width * height:
        width, height = round_to_multiple(src_w), round_to_multiple(src_h)
    return torch.stack([torch.from_numpy(np.asarray(Image.fromarray(f[..., :3]).resize((width, height), Image.Resampling.LANCZOS)).copy()) for f in frames])
