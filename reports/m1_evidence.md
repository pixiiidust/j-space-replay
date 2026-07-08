# M1 evidence (issue #3)

Run on Pixie (RTX 5070 Ti, torch 2.11.0+cu128, transformers 5.13, NF4+SDPA), 2026-07-08.

## Timing criterion: 15 s clip → valid trace ≤ 90 s

`fixtures/clips/_timing_traffic_15s.mp4` (15 s, 8 frame groups, 1600 visual tokens):
**19.8 s wall-clock including model load**; pipeline stages: sampling 0.11 s,
prefill+generate 5.04 s, lens decode 1.22 s. Trace passes `validate_trace`.

## Risk gate (day-4): frame-group × layer grid — VERDICT: STRUCTURE, PROCEED

Method note: mean-pooling patch residuals per group before unembedding washes
signal out (grid static junk at all layers). Per-patch decode + patch-share
aggregation preserves it; the pipeline now does the latter.

`ball_drop.mp4`, "Why does the ball fall?", wordlike-filtered top-3 per cell
(`scripts/render_grid.py`):

```
layer | g0 0-2s            | g1 2-4s            | g2 4-6s            | g3 6-8s
    0 | ·                  | ·                  | ·                  | okable
    4 | registrazione,Guid | registrazione      | ascript,okable     | okable,ascript
    8 | longleftrightarrow | registrazione,long | registrazione,IFn, | vla,MLElement
   12 | longleftrightarrow | registrazione,ummi | ummings,registrazi | registrazione
   16 | GuidId             | GuidId             | GuidId,UsageId     | GuidId
   20 | ·                  | ·                  | ·                  | vla
   24 | GuidId             | GuidId             | GuidId             | ibrator
   27 | gray,the,grey      | brown,red,on       | the,brown,ball     | the,brown,Brown
```

Per-patch top-1 frequency probe (`scripts/patch_probe.py`), layers 26–27:
clear content tokens with temporal differentiation —
g0-g1: `brown` (table), `red` (ball), `background`, `rolling`;
g2 (ball falls at ~4 s): `move ×5`, `left ×3`, `ball ×5`, `balance`;
g3: `brown ×19`, `bar`, `moves`. Junk share drops from ~85 % (mid layers)
to ~40 % (layer 27).

Reading: signal on visual tokens **exists and is time-localized** in late
layers (24–27); mid layers are junk under the raw logit lens. This matches the
"misaligned basis" pattern the PRD anticipated — the J-lens port (#8) is the
expected cleanup for mid layers, **not** a blocker for v0.1. Answer-token lens
is fully coherent: late layers converge on the emitted token, mid layers show
workspace-like precursors (e.g. `是因为`/"because" at L16–20 before `because`
is emitted; `gravity` readable at L24 several tokens early).

## Tests

`uv run pytest -q` → 18 passed (14 pure-logic: sampler/grouping/position
map/schema; 4 GPU golden: 3 fixture-clip shape locks + answer-token alignment).
