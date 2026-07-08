"""Risk-gate diagnostic: per-patch (not mean-pooled) lens readouts on visual tokens.

For each frame group and a few layers, decode EVERY patch token and aggregate
top-1 tokens by frequency, split into 'wordlike' vs junk. Structure here =
signal exists on visual tokens; static junk everywhere = escalate per PRD tree.

    uv run python scripts/patch_probe.py fixtures/clips/ball_drop.mp4 "Why does the ball fall?"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import torch

from jsr.capture import ResidualCapture
from jsr.lens import lens_topk
from jsr.model import MAX_PIXELS, decoder_layers, load_model_and_processor
from jsr.positions import build_position_map
from jsr.video import group_frames, sample_frames

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 console

PROBE_LAYERS = [4, 8, 12, 16, 20, 22, 24, 26, 27]


def wordlike(tok: str) -> bool:
    s = tok.strip()
    return len(s) >= 2 and s.isascii() and s.replace("-", "").isalpha()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("question", nargs="?", default="Describe what happens in this video.")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    from qwen_vl_utils import process_vision_info

    model, processor = load_model_and_processor()
    sampled = sample_frames(args.clip, fps=1.0)
    groups = group_frames(sampled)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": sampled.frames, "max_pixels": MAX_PIXELS,
         "sample_fps": sampled.sample_fps},
        {"type": "text", "text": args.question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, videos, vk = process_vision_info(messages, return_video_kwargs=True)
    fps_kw = vk.get("fps")
    if isinstance(fps_kw, list):
        fps_kw = fps_kw[0]
    inputs = processor(text=[text], videos=videos, return_tensors="pt", fps=float(fps_kw)).to(
        model.device
    )
    with ResidualCapture(decoder_layers(model)) as cap:
        with torch.inference_mode():
            model(**inputs)  # prefill only — no generation needed for this probe
    prefill = cap.prefill_stack()

    pm = build_position_map(
        inputs.input_ids[0].tolist(),
        tuple(inputs.video_grid_thw[0].tolist()),
        getattr(processor.image_processor, "merge_size", 2),
        model.config.video_token_id,
    )
    tok = processor.tokenizer
    for layer in PROBE_LAYERS:
        print(f"\n=== layer {layer} ===")
        for g in groups:
            idxs = pm.group_token_indices(g.group)
            ids, _ = lens_topk(model, prefill[layer, idxs, :], k=1)
            strings = [tok.decode([i]) for i in ids[:, 0].tolist()]
            words = Counter(s.strip() for s in strings if wordlike(s))
            junk_share = 1 - sum(words.values()) / len(strings)
            top = ", ".join(f"{w}x{c}" for w, c in words.most_common(args.top))
            print(f"  g{g.group} {g.time_start:.0f}-{g.time_end:.0f}s junk={junk_share:.0%}: {top}")


if __name__ == "__main__":
    main()
