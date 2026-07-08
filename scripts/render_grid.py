"""Render the frame-group x layer logit-lens grid from a trace.json as text.

The M1 risk-gate artifact: structure (early echo -> mid workspace -> late answer
drift across layers/time) means the lens sees signal on visual tokens.

    uv run python scripts/render_grid.py trace.json [--top 2] [--layer-step 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 console


def wordlike(tok: str) -> bool:
    s = tok.strip()
    return len(s) >= 2 and s.isascii() and s.replace("-", "").isalpha()


def cell(readout: dict, top: int, words_only: bool) -> str:
    toks = [t.strip() for t in readout["top_tokens"]]
    if words_only:
        toks = [t for t in toks if wordlike(t)]
    toks = [t.replace("\n", "\\n") or "·" for t in toks[:top]]
    return ",".join(toks)[:18] if toks else "·"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--raw", action="store_true", help="show all tokens, not just wordlike ones")
    args = ap.parse_args()

    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    groups = trace["frame_groups"]
    n_layers = trace["meta"]["n_layers"]
    layers = list(range(0, n_layers, args.layer_step))
    if layers[-1] != n_layers - 1:
        layers.append(n_layers - 1)

    print(f"clip={trace['video_id']}  q={trace['question']!r}")
    print(f"answer: {trace['answer']}\n")
    header = "layer | " + " | ".join(
        f"g{g['group']} {g['time_start']:.0f}-{g['time_end']:.0f}s".ljust(18) for g in groups
    )
    print(header)
    print("-" * len(header))
    by_layer = {
        g["group"]: {r["layer"]: r for r in g["raw_readouts"]} for g in groups
    }
    for layer in layers:
        row = f"{layer:5d} | " + " | ".join(
            cell(by_layer[g["group"]][layer], args.top, not args.raw).ljust(18) for g in groups
        )
        print(row)


if __name__ == "__main__":
    main()
