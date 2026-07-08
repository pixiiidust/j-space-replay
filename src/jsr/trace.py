"""Trace pipeline: video + question -> trace.json (SPEC schema v1).

    uv run python -m jsr.trace fixtures/clips/ball_drop.mp4 "What happens?" -o trace.json

Stages (also the progress stages M3's job runner reports):
  sampling -> prefill+capture -> generating -> lens decode -> write
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from jsr.capture import ResidualCapture
from jsr.lens import decode_tokens, lens_topk
from jsr.model import DEFAULT_QUESTION, MAX_PIXELS, decoder_layers, load_model_and_processor
from jsr.positions import build_position_map
from jsr.schema import SCHEMA_VERSION, validate_trace
from jsr.video import group_frames, sample_frames


def run_trace(
    clip: str | Path,
    question: str = DEFAULT_QUESTION,
    *,
    model,
    processor,
    fps: float = 1.0,
    max_pixels: int = MAX_PIXELS,
    top_k: int = 15,
    answer_top_k: int = 5,
    max_new_tokens: int = 128,
    on_stage=None,
) -> dict:
    from qwen_vl_utils import process_vision_info

    timings: dict[str, float] = {}

    def _stage(name):
        if on_stage:
            on_stage(name)
        timings[name] = time.perf_counter()

    def _done(name):
        timings[name] = round(time.perf_counter() - timings[name], 2)

    clip = Path(clip)
    _stage("sampling")
    sampled = sample_frames(clip, fps=fps)
    groups = group_frames(sampled)
    _done("sampling")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sampled.frames,
                    "max_pixels": max_pixels,
                    "sample_fps": sampled.sample_fps,
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    fps_kw = video_kwargs.get("fps")
    if isinstance(fps_kw, list):
        fps_kw = fps_kw[0] if fps_kw else None
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **({"fps": float(fps_kw)} if fps_kw is not None else {}),
    ).to(model.device)

    _stage("prefill_and_generate")
    with ResidualCapture(decoder_layers(model)) as cap:
        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    cap.assert_no_gpu_accumulation()
    _done("prefill_and_generate")

    input_ids = inputs.input_ids[0].tolist()
    answer_ids = out_ids[0][len(input_ids):].tolist()
    tokenizer = processor.tokenizer
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    grid = tuple(inputs.video_grid_thw[0].tolist())
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    pm = build_position_map(input_ids, grid, merge_size, model.config.video_token_id)
    n_temporal_groups = pm.grid[0]
    if n_temporal_groups != len(groups):
        raise RuntimeError(
            f"temporal grid {n_temporal_groups} != frame-group bookkeeping {len(groups)}"
        )

    _stage("lens_decode")
    prefill = cap.prefill_stack()  # (L, seq, d)
    steps = cap.steps_stack()  # (L, n_steps, d)
    n_layers, _, d = prefill.shape

    # frame-group x layer readouts: decode EVERY patch token, aggregate top-3
    # tokens by the share of the group's patches that read them. Mean-pooling
    # the residuals first washes signal out (verified on ball_drop at the risk
    # gate: per-patch shows brown/red/ball at late layers; pooled shows junk).
    vis = prefill[:, pm.visual_indices, :]  # (L, V, d)
    n_visual = vis.shape[1]
    patch_ids, _ = lens_topk(model, vis.reshape(-1, d), k=3, chunk=1024)
    patch_ids = patch_ids.reshape(n_layers, n_visual, 3)
    per_group = pm.grid[1] * pm.grid[2]

    frame_groups = []
    for gi, g in enumerate(groups):
        lo, hi = gi * per_group, (gi + 1) * per_group
        readouts = []
        for layer in range(n_layers):
            ids_gl = patch_ids[layer, lo:hi]  # (P, 3), rows have distinct vocab ids
            counts = torch.bincount(ids_gl.reshape(-1)).float()
            share, tok_ids = counts.topk(min(top_k, int((counts > 0).sum())))
            share = share / per_group
            readouts.append(
                {
                    "layer": layer,
                    "top_tokens": [tokenizer.decode([t]) for t in tok_ids.tolist()],
                    "strengths": [round(v, 4) for v in share.tolist()],
                    "patch_top1": patch_ids[layer, lo:hi, 0].tolist(),
                }
            )
        frame_groups.append(
            {
                "group": g.group,
                "time_start": round(g.time_start, 3),
                "time_end": round(g.time_end, 3),
                "frame_indices": g.frame_indices,
                "n_patch_tokens": per_group,
                "patch_grid": [pm.grid[1], pm.grid[2]],
                "raw_readouts": readouts,
                "concepts": [],  # filled by M2 label extraction
            }
        )

    # answer-token x layer: residual that PRODUCED token i lives at the previous
    # position — prefill's last position for token 0, step i-1 for token i.
    answer_tokens = []
    n_producing = min(len(answer_ids), (steps.shape[1] if steps.numel() else 0) + 1)
    if n_producing:
        producing = torch.cat(
            [prefill[:, -1:, :], steps[:, : n_producing - 1, :]] if n_producing > 1
            else [prefill[:, -1:, :]],
            dim=1,
        )  # (L, n_producing, d)
        a_ids, a_vals = lens_topk(model, producing.permute(1, 0, 2).reshape(-1, d), k=answer_top_k)
        a_tokens = decode_tokens(tokenizer, a_ids)
        for i in range(n_producing):
            by_layer = {}
            for layer in range(n_layers):
                row = i * n_layers + layer
                by_layer[str(layer)] = {
                    "top_tokens": a_tokens[row],
                    "strengths": [round(v, 3) for v in a_vals[row].tolist()],
                }
            answer_tokens.append(
                {"token": tokenizer.decode([answer_ids[i]]), "readouts_by_layer": by_layer}
            )
    _done("lens_decode")

    trace = {
        "schema": SCHEMA_VERSION,
        "video_id": clip.stem,
        "question": question,
        "answer": answer,
        "meta": {
            "model": "qwen2.5-vl-7b-instruct-bnb-nf4",
            "lens": "logit-lens-v1",
            "temporal_resolution_frames": 2,
            "strength_normalization": "patch-share-top3 (visual) / raw-logit (answer)",
            "n_layers": n_layers,
            "fps": fps,
            "max_pixels": max_pixels,
            "sampled_frames": len(sampled.frames),
            "video_grid_thw": list(grid),
            "input_tokens": len(input_ids),
            "visual_tokens": len(pm.visual_indices),
            "timings_s": timings,
        },
        "frame_groups": frame_groups,
        "answer_tokens": answer_tokens,
        "grounding": [],  # filled by M3 grounding queries
    }
    validate_trace(trace)
    return trace


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m jsr.trace", description=__doc__)
    ap.add_argument("clip")
    ap.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    ap.add_argument("-o", "--out", default="trace.json")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, processor = load_model_and_processor()
    trace = run_trace(
        args.clip,
        args.question,
        model=model,
        processor=processor,
        fps=args.fps,
        max_pixels=args.max_pixels,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        on_stage=lambda s: print(f"[stage] {s}", flush=True),
    )
    Path(args.out).write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
    total = time.perf_counter() - t0
    print(f"answer: {trace['answer']}")
    print(f"timings: {trace['meta']['timings_s']} | total incl. model load {total:.1f}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
