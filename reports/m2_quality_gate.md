# M2 quality gate (issue #4) — 2026-07-08

Method: 10 known-content synthetic clips (3 fixtures + 7 gate clips), full
pipeline (trace → caption pass → candidate vocab → z-scored patch-share
matching, layers ≥ 20, baseline v2 = per-layer + per-common-token stats over a
20-clip corpus). Judgments are against the clips' known generated content.
Raw per-clip output: `reports/m2_quality_gate.json`.

| clip | known content | concepts shown (z) | sensible |
|---|---|---|---|
| ball_drop | red ball rolls on brown table, falls | brown(3.3), ball(1.0), ball brown(1.0) | 3/3 |
| shape_morph | blue square appears, grows, turns green | square(3.2), blue(3.1), smaller(2.1)✗, background(1.2) | 3/4 |
| traffic | light red→green, gray car moves | light(4.4), traffic(2.6), traffic light(2.6), black(1.8), gray(1.3) | 5/5 |
| blue_circle_drop | blue circle drops, white bg | background(3.4), blue(2.0) | 2/2 |
| green_square_slide | green square slides, dark bg | black(4.1), green(3.3), background(1.8) | 3/3 |
| red_square_grow | red square grows, white bg | background(2.2) | 1/1 |
| yellow_circle_slide | yellow circle slides, white bg | background(2.2) | 1/1 |
| purple_square_drop | purple square drops, dark bg | black(1.3), purple(1.3), background(1.2), square(0.8) | 4/4 |
| green_circle_grow | green circle grows, dark bg | background(1.2) | 1/1 |
| blue_square_slide | blue square slides, white bg | blue(1.7), background(1.7) | 2/2 |

**Precision: 24/26 ≈ 92 % of shown concepts judged sensible.**
**Recall: no clip yields ≥ 6 concepts → the "≥ 6 of top-10 per clip" gate FAILS on recall.**

## Verdict → fallback invoked (per issue #4, non-blocking)

- Concept labels are marked **experimental** (`meta.concepts_quality`); the
  **raw-token grid stays the primary view** (the UI already renders
  concept-less traces via the wordlike-token fallback).
- Known cause of low recall, recorded honestly: (1) the logit lens only reads
  content reliably at layers ≥ 20 (M1 risk-gate finding) — fewer cells to draw
  from; (2) the synthetic baseline corpus shares the gate clips' color palette,
  so exactly those color words get normalized toward 0 (z-scoring correctly
  highlights what is *distinctive vs corpus*, and the corpus is too similar).
  A natural-video baseline post-v0.1 and the J-lens (#8) mid-layer cleanup are
  the expected recall fixes.
- What did NOT happen: junk labels. The failure mode the milestone feared
  ("`▁the oor ▁of` in a grid") is absent — candidate-vocabulary matching plus
  per-token baseline stats eliminated lens-furniture tokens entirely.
