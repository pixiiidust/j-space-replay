"""Build per-layer baseline stats for z-score normalization (M2).

Generates ~20 varied synthetic clips (colors x shapes x motions x backgrounds),
traces each, and fits mean/std of the patch-share of *wordlike* readout tokens
per layer. Output: src/jsr/baseline_stats.json (committed — ships with the app).

Honesty note (recorded in the JSON): the baseline corpus is synthetic 2-D
animation, not natural video. Good enough for demo-quality normalization;
revisit with natural clips post-v0.1.

    uv run python scripts/make_baseline.py [--out src/jsr/baseline_stats.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw

import sys

sys.path.insert(0, str(Path(__file__).parent))
from make_fixtures import FPS, H, W, write_clip  # noqa: E402

from jsr.labels import canon, wordlike  # noqa: E402
from jsr.model import load_model_and_processor  # noqa: E402
from jsr.trace import run_trace  # noqa: E402

COLORS = {"red": (200, 30, 30), "blue": (30, 60, 200), "green": (30, 170, 60),
          "yellow": (220, 200, 40), "purple": (140, 40, 180)}
BACKGROUNDS = {"white": (247, 247, 245), "dark": (40, 40, 44)}


def scene(color, shape, motion, bg):
    def gen(n):
        for i in range(n):
            t = i / FPS
            img = Image.new("RGB", (W, H), BACKGROUNDS[bg])
            d = ImageDraw.Draw(img)
            if motion == "slide":
                x, y, r = 80 + (W - 300) * (t / (n / FPS)), H // 2, 50
            elif motion == "drop":
                x, y, r = W // 2, min(80 + 380 * t * t / 16, H - 90), 50
            else:  # grow
                x, y, r = W // 2, H // 2, 20 + 70 * min(t / (n / FPS) * 1.5, 1.0)
            if shape == "circle":
                d.ellipse([x - r, y - r, x + r, y + r], fill=COLORS[color])
            else:
                d.rectangle([x - r, y - r, x + r, y + r], fill=COLORS[color])
            yield __import__("numpy").asarray(img)

    return gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="defaults to the lens-specific stats file")
    ap.add_argument("--seconds", type=int, default=6)
    ap.add_argument("--lens", choices=["logit-lens-v1", "j-lens-v1"], default="logit-lens-v1")
    args = ap.parse_args()

    from jsr.labels import _BASELINE_PATHS  # lens -> shipped stats path

    if args.out is None:
        args.out = str(_BASELINE_PATHS[args.lens])
    jlens = None
    if args.lens == "j-lens-v1":
        from jsr.lens import JLens

        jlens = JLens.load()

    combos = list(itertools.product(COLORS, ["circle", "square"], ["slide", "drop", "grow"],
                                    BACKGROUNDS))[::3][:20]  # spread over the grid, 20 clips
    clip_dir = Path("fixtures/clips/baseline")
    clip_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor()
    per_layer: dict[int, list[float]] = {}
    per_token: dict[int, dict[str, list[float]]] = {}
    n_cells = 0
    for i, (color, shape, motion, bg) in enumerate(combos):
        path = clip_dir / f"baseline_{i:02d}_{color}_{shape}_{motion}_{bg}.mp4"
        if not path.exists():
            write_clip(path, args.seconds, scene(color, shape, motion, bg))
        trace = run_trace(path, model=model, processor=processor, jlens=jlens)
        for g in trace["frame_groups"]:
            n_cells += 1
            for r in g["raw_readouts"]:
                for tok, share in zip(r["top_tokens"], r["strengths"]):
                    if wordlike(tok):
                        per_layer.setdefault(r["layer"], []).append(share)
                        per_token.setdefault(r["layer"], {}).setdefault(canon(tok), []).append(share)
        print(f"[{i + 1}/{len(combos)}] {path.name}: answer={trace['answer'][:60]!r}")

    layers = {
        str(layer): {
            "mean": round(statistics.mean(v), 6),
            "std": round(statistics.pstdev(v), 6),
            "n": len(v),
        }
        for layer, v in sorted(per_layer.items())
    }
    # tokens that read out in >= `common_frac` of baseline cells at a layer are
    # lens furniture, not content: give them their own (higher) baseline so
    # their z-scores collapse unless a clip truly exceeds their usual share
    common_frac = 0.15
    min_cells = max(2, int(n_cells * common_frac))
    common_tokens = {
        str(layer): {
            tok: {
                "mean": round(statistics.mean(v), 6),
                "std": round(max(statistics.pstdev(v), 1e-4), 6),
                "cells": len(v),
            }
            for tok, v in toks.items()
            if len(v) >= min_cells
        }
        for layer, toks in sorted(per_token.items())
    }
    out = {
        "_note": "v2: per-layer wordlike patch-share stats + per-(layer, canonical-token) stats "
                 f"for tokens present in >= {common_frac:.0%} of baseline cells. Synthetic 2-D "
                 f"baseline corpus ({len(combos)} clips, {n_cells} cells); demo-quality.",
        "version": 2,
        "lens": args.lens,
        "layers": layers,
        "common_tokens": common_tokens,
    }
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    n_common = sum(len(v) for v in common_tokens.values())
    print(f"wrote {args.out} ({len(layers)} layers, {n_common} common-token entries)")


if __name__ == "__main__":
    main()
