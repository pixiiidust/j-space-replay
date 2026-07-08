"""Generate the three synthetic fixture clips (deterministic, known content).

Synthetic clips instead of downloaded footage: content is exactly known (good for
golden tests and hand-labelling), generation is reproducible, and nothing
copyrighted lands in the repo. Frame pixels are deterministic; encoded bytes may
differ across ffmpeg builds, but frame count / resolution / timing do not.

    uv run python scripts/make_fixtures.py [-o fixtures/clips]

Clips (all 480p = 854x480, 24 fps, H.264 yuv420p):
  ball_drop.mp4   (8 s)  red ball rolls right along a table edge, falls off at ~4 s, bounces
  shape_morph.mp4 (10 s) blue square appears at 2 s, grows, turns green at 6 s
  traffic.mp4     (12 s) traffic light red -> green at 5 s; grey car starts moving after
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

W, H, FPS = 854, 480, 24
BG = (247, 247, 245)


def _frames_ball_drop(n: int):
    table_y = 300
    for i in range(n):
        t = i / FPS
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([60, table_y, 560, table_y + 18], fill=(120, 84, 50))  # table
        r = 28
        if t < 4.0:  # rolling right along the table
            x = 90 + (520 - 90) * (t / 4.0)
            y = table_y - r
        else:  # off the edge: projectile fall with one bounce at the floor
            dt = t - 4.0
            x = 520 + 120 * dt
            floor = H - 40 - r
            y = (table_y - r) + 420 * dt * dt
            if y > floor:  # crude bounce
                yb = y - floor
                y = floor - min(yb * 0.6, 90)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(200, 30, 30))
        d.rectangle([0, H - 40, W, H], fill=(200, 200, 194))  # floor
        yield np.asarray(img)


def _frames_shape_morph(n: int):
    for i in range(n):
        t = i / FPS
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        if t >= 2.0:
            size = 40 + min((t - 2.0) / 4.0, 1.0) * 140
            color = (30, 60, 200) if t < 6.0 else (30, 170, 60)
            cx, cy = W // 2, H // 2
            d.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
        yield np.asarray(img)


def _frames_traffic(n: int):
    for i in range(n):
        t = i / FPS
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 340, W, 480], fill=(90, 90, 90))  # road
        d.rectangle([640, 80, 700, 260], fill=(40, 40, 40))  # signal pole box
        green = t >= 5.0
        d.ellipse([650, 95, 690, 135], fill=(220, 40, 40) if not green else (70, 30, 30))
        d.ellipse([650, 200, 690, 240], fill=(40, 200, 70) if green else (25, 70, 35))
        car_x = 80 if not green else 80 + (t - 5.0) * 90
        d.rectangle([car_x, 360, car_x + 150, 420], fill=(70, 70, 80))  # car body
        d.ellipse([car_x + 15, 405, car_x + 55, 445], fill=(20, 20, 20))
        d.ellipse([car_x + 95, 405, car_x + 135, 445], fill=(20, 20, 20))
        yield np.asarray(img)


CLIPS = {
    "ball_drop.mp4": (8, _frames_ball_drop),
    "shape_morph.mp4": (10, _frames_shape_morph),
    "traffic.mp4": (12, _frames_traffic),
}


def write_clip(path: Path, seconds: int, gen) -> None:
    n = seconds * FPS
    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        for arr in gen(n):
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="fixtures/clips")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (seconds, gen) in CLIPS.items():
        path = out / name
        write_clip(path, seconds, gen)
        print(f"wrote {path} ({seconds}s, {seconds * FPS} frames, {path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
