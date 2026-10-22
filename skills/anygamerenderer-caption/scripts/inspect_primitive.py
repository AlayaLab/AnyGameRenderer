"""Inspect a crude (white-model) recording: which proxy colours it contains, roughly how many proxies of each, the clip
length / frame count AnyGameRenderer will generate, and a contact sheet of sampled frames to look at the proxy shape and camera.

Usage: python inspect_primitive.py clip/primitive.mp4 [--sheet clip/primitive_sheet.jpg]
"""
import argparse, math
import cv2, numpy as np

CANON = {'red': 0, 'orange': 13, 'yellow': 27, 'yellow-green': 40, 'green': 60, 'cyan': 92, 'blue': 113, 'purple': 135, 'magenta': 152}  # OpenCV hue (0-180)


def hue_name(h):
    return min(CANON, key=lambda c: min(abs(CANON[c] - h), 180 - abs(CANON[c] - h)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('video'); ap.add_argument('--sheet', default=None); a = ap.parse_args()
    cap = cv2.VideoCapture(a.video); n = int(cap.get(7)); fps = cap.get(5) or 24; w, h = int(cap.get(3)), int(cap.get(4))
    dur = n / fps; gen = min(345, 5 + ((int(dur * 24 + 1e-6) - 5) // 17) * 17)
    print(f'{a.video}: {w}x{h}, {fps:.3g} fps, {n} frames = {dur:.2f} s -> model generates {gen} frames ({gen / 24:.2f} s)')
    counts, areas, frames, pinned = {}, {}, [], {}
    idx = sorted(set(int(i) for i in np.linspace(0, n - 1, 12)))
    for i in idx:
        cap.set(1, i); ok, f = cap.read()
        if not ok: continue
        frames.append((i, f)); hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 1] > 100) & (hsv[..., 2] > 60)).astype(np.uint8)  # saturated = proxy; grey geometry is excluded
        num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for k in range(1, num):
            if stats[k, cv2.CC_STAT_AREA] < 25: continue
            hh = int(np.median(hsv[..., 0][lab == k])); name = hue_name(hh)
            x, y, bw, bh, area = stats[k]
            counts.setdefault(name, []).append(1); areas.setdefault(name, []).append(area)
            if area > 0.03 * w * h and y + bh >= 0.95 * h and x + bw / 2 > 0.6 * w: pinned[name] = pinned.get(name, 0) + 1  # 大块、贴着下缘、偏右: 第一人称武器的特征
    if not counts:
        print('no saturated proxy colours found — is this really a white-model video?')
    for name in sorted(counts, key=lambda c: -sum(areas[c])):
        per = len(counts[name]) / len(frames); mean_area = np.mean(areas[name]); big = max(areas[name])
        print(f'  {name:13s} ~{per:4.1f} blobs per sampled frame (touching proxies merge into one blob), mean area {mean_area:7.0f} px, largest {big:7.0f} px'
              + ('   <- big blob pinned to the lower right in most frames: probably the first-person weapon ("many": false)' if pinned.get(name, 0) >= 0.8 * len(frames) else '')
              + ('   <- several per frame: a group ("many": true)' if per >= 2 else ''))
    print('shape (box / wire / cylinder / sphere) and camera: look at the contact sheet.')
    if frames:
        tw, th = 336, int(336 * h / w); cols = 4; rows = math.ceil(len(frames) / cols)
        sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
        for j, (i, f) in enumerate(frames):
            t = cv2.resize(f, (tw, th)); cv2.putText(t, f'f{i}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            sheet[(j // cols) * th:(j // cols + 1) * th, (j % cols) * tw:(j % cols + 1) * tw] = t
        out = a.sheet or a.video.rsplit('.', 1)[0] + '_sheet.jpg'; cv2.imwrite(out, sheet); print('contact sheet:', out)


if __name__ == '__main__':
    main()
