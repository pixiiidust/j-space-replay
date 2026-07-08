"""Characterize which layers show workspace-like readouts, from trace.json files.

Ported idea from jlens-qwen36's workspace_range.py, adapted to video traces:

- visual patches per layer: wordlike share of patch top-1 readouts (content
  proxy) vs junk share — the mid-layer signal the J-lens is supposed to unlock.
- answer positions per layer: echo (reads the previous emitted token),
  motor (reads the token about to be emitted), workspace (wordlike, neither),
  junk (non-wordlike).

Feeds two decisions: the j-lens concept layer floor (jsr.labels.LAYER_FLOOR)
and the 6-8 display layers for the UI. Compare lenses by tracing the same clip
with --lens logit-lens-v1 and j-lens-v1 first.

    uv run python scripts/workspace_range.py trace_logit.json trace_jlens.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def wordlike(tok: str) -> bool:
    s = tok.strip()
    return len(s) >= 2 and s.isascii() and s.replace("-", "").isalpha()


def norm(s: str) -> str:
    return s.strip().lower()


def analyze(path: Path) -> None:
    trace = json.loads(path.read_text(encoding="utf-8"))
    n_layers = trace["meta"]["n_layers"]
    strings = trace["meta"]["token_strings"]

    vis_word = [0] * n_layers
    vis_total = [0] * n_layers
    for g in trace["frame_groups"]:
        for r in g["raw_readouts"]:
            for tid in r["patch_top1"]:
                vis_word[r["layer"]] += wordlike(strings.get(str(tid), ""))
                vis_total[r["layer"]] += 1

    ans = [{"echo": 0, "motor": 0, "workspace": 0, "junk": 0} for _ in range(n_layers)]
    tokens = [a["token"] for a in trace["answer_tokens"]]
    for i, a in enumerate(trace["answer_tokens"]):
        prev = tokens[i - 1] if i else ""
        for layer_s, r in a["readouts_by_layer"].items():
            top = r["top_tokens"][0]
            cls = ("motor" if norm(top) == norm(a["token"])
                   else "echo" if prev and norm(top) == norm(prev)
                   else "workspace" if wordlike(top) else "junk")
            ans[int(layer_s)][cls] += 1

    print(f"\n== {path.name}  lens={trace['meta']['lens']}  clip={trace['video_id']} ==")
    print(f"{'layer':>5} | {'visual wordlike':>15} | {'ans echo':>8} {'motor':>6} "
          f"{'workspace':>9} {'junk':>5}")
    n_ans = max(1, len(tokens))
    for layer in range(n_layers):
        vw = vis_word[layer] / max(1, vis_total[layer])
        a = ans[layer]
        print(f"{layer:5d} | {vw:15.0%} | {a['echo'] / n_ans:8.0%} {a['motor'] / n_ans:6.0%} "
              f"{a['workspace'] / n_ans:9.0%} {a['junk'] / n_ans:5.0%}")

    # suggested display band: the 8 contiguous layers with the highest
    # combined visual-wordlike + answer-workspace share
    score = [vis_word[layer] / max(1, vis_total[layer]) + ans[layer]["workspace"] / n_ans
             for layer in range(n_layers)]
    best = max(range(n_layers - 7), key=lambda s: sum(score[s : s + 8]))
    print(f"suggested display band: layers {best}-{best + 7}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    args = ap.parse_args()
    for t in args.traces:
        analyze(Path(t))


if __name__ == "__main__":
    main()
