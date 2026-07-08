"""Generate golden expected-shape fixtures for the 3 test clips.

Golden files lock the *structural* contract (token counts, grid, timing map) —
not model text output, which legitimately varies with sampling/library versions.

    uv run python scripts/make_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

from jsr.model import DEFAULT_QUESTION, load_model_and_processor
from jsr.schema import trace_shape
from jsr.trace import run_trace

CLIPS = ["ball_drop.mp4", "shape_morph.mp4", "traffic.mp4"]


def main() -> None:
    out_dir = Path("fixtures/golden")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model_and_processor()
    for clip in CLIPS:
        trace = run_trace(
            Path("fixtures/clips") / clip, DEFAULT_QUESTION, model=model, processor=processor
        )
        golden = trace_shape(trace)
        path = out_dir / f"{Path(clip).stem}.golden.json"
        path.write_text(json.dumps(golden, indent=2), encoding="utf-8")
        Path("reports").mkdir(exist_ok=True)
        (Path("reports") / f"trace_{Path(clip).stem}_default.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"wrote {path}: {golden['n_groups']} groups x {golden['n_layers']} layers, "
              f"{golden['visual_tokens']} visual tokens")


if __name__ == "__main__":
    main()
