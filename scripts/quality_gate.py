"""M2 quality gate: label 10 known-content clips, dump top-10 concepts per clip.

The gate (issue #4): >= 6 of the top-10 concepts per clip judged sensible on at
least... (hand-judged against the clips' known synthetic content). This script
produces the evidence table; the judgment column is filled by a human/reviewer.

    uv run python scripts/quality_gate.py [-o reports/m2_quality_gate.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_baseline import scene  # noqa: E402
from make_fixtures import write_clip  # noqa: E402

from jsr.labels import add_concepts  # noqa: E402
from jsr.model import load_model_and_processor  # noqa: E402
from jsr.trace import run_trace  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (clip file | generator spec, known content description)
GATE_CLIPS: list[tuple[str, dict | None, str]] = [
    ("ball_drop.mp4", None, "red ball rolls right along brown table, falls off edge ~4s, bounces on gray floor"),
    ("shape_morph.mp4", None, "blue square appears at 2s on white bg, grows, turns green at 6s"),
    ("traffic.mp4", None, "traffic light red->green at 5s; gray car starts moving right"),
    ("gate_blue_circle_drop.mp4", dict(color="blue", shape="circle", motion="drop", bg="white"), "blue circle drops from top, white background"),
    ("gate_green_square_slide.mp4", dict(color="green", shape="square", motion="slide", bg="dark"), "green square slides left-to-right, dark background"),
    ("gate_red_square_grow.mp4", dict(color="red", shape="square", motion="grow", bg="white"), "red square grows in center, white background"),
    ("gate_yellow_circle_slide.mp4", dict(color="yellow", shape="circle", motion="slide", bg="white"), "yellow circle slides left-to-right, white background"),
    ("gate_purple_square_drop.mp4", dict(color="purple", shape="square", motion="drop", bg="dark"), "purple square drops from top, dark background"),
    ("gate_green_circle_grow.mp4", dict(color="green", shape="circle", motion="grow", bg="dark"), "green circle grows in center, dark background"),
    ("gate_blue_square_slide.mp4", dict(color="blue", shape="square", motion="slide", bg="white"), "blue square slides left-to-right, white background"),
]


def top10_concepts(trace: dict) -> list[dict]:
    best: dict[str, dict] = {}
    for g in trace["frame_groups"]:
        for c in g["concepts"]:
            prev = best.get(c["label"])
            if prev is None or c["strength"] > prev["strength"]:
                best[c["label"]] = {**c, "group": g["group"]}
    return sorted(best.values(), key=lambda c: -c["strength"])[:10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=None, help="defaults per lens")
    ap.add_argument("--lens", choices=["logit-lens-v1", "j-lens-v1"], default="logit-lens-v1")
    args = ap.parse_args()
    if args.out is None:
        args.out = ("reports/m2_quality_gate.json" if args.lens == "logit-lens-v1"
                    else "reports/m2_quality_gate_jlens.json")
    jlens = None
    if args.lens == "j-lens-v1":
        from jsr.lens import JLens

        jlens = JLens.load()

    clip_dir = Path("fixtures/clips")
    model, processor = load_model_and_processor()
    results = []
    for name, spec, known in GATE_CLIPS:
        path = clip_dir / name
        if not path.exists():
            assert spec is not None, f"missing fixture {name}"
            write_clip(path, 6, scene(**spec))
        trace = run_trace(path, model=model, processor=processor, jlens=jlens)
        add_concepts(trace, model=model, processor=processor, clip=path)
        top = top10_concepts(trace)
        results.append({
            "clip": name,
            "known_content": known,
            "answer": trace["answer"][:200],
            "caption": trace["meta"].get("caption", "")[:200],
            "top10": [
                {"label": c["label"], "strength": c["strength"], "layer": c["layer"],
                 "group": c["group"], "source_tokens": c["source_tokens"]}
                for c in top
            ],
        })
        print(f"{name}: " + ", ".join(f"{c['label']}({c['strength']})" for c in top))

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
