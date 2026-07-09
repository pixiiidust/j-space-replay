# J-lens port evidence (issue #8)

Run on Pixie (RTX 5070 Ti, torch 2.11.0+cu128, transformers 5.13, NF4+SDPA),
2026-07-08. Lens fit by `scripts/fit_jlens.py` over 15 video prompts
(3 fixtures + 7 gate clips + 5 baseline clips, default question, prefill
positions, skip_first=4); ~177 s/layer, ~80 min total on the 5070 Ti.

Two fits were run. The first followed the jlens-qwen36 repo's chain seed
(final-norm Jacobian at the top). Re-reading the workspace paper
(transformer-circuits.pub/2026/workspace) against the repo showed the repo
deviates from the paper: the paper defines J on the PRE-norm final residual
with normalization only at read time — "the logit lens ... corresponds to
setting J_l = I in our formulation" — so the final layer's J is the
identity. The corrected (identity-seed) fit is `j-lens-v1`; the norm-seeded
variant is kept at `jlens/j_lens_v1_normseed.pt` for the A/B below.

## Component verification (criterion 1: match autograd, ~1e-3 fp32)

Ground truth = torch autograd through the true-weight rebuilt layer (itself
proven bit-exact against the NF4 runtime in Gate 0 when fed the runtime's own
dequantized weights). Real captured video-prefill activations, layers 0/12/27:

| component | rel. Frobenius error |
|---|---|
| RMSNorm Jacobian (closed form) | ~7e-8 |
| SwiGLU MLP branch (Hadamard, per-position norm fold) | ~2.5e-6 |
| Attention branch (W_o cotangents, mRoPE+GQA, exact S=64) | 6e-7 – 8.7e-6 |
| Attention branch (random cotangent rows, full S=827) | 1e-6 – 4e-6 |
| Full layer (known junction approximation), layers 1–27 | 0.9% – 3.6% |
| Full layer 0 (excluded from chain: M_0 unused) | 15% |

**PASS** — every analytic component matches autograd well below 1e-3; the
full-layer residual is the reference repo's documented branch-product
junction approximation at the same magnitude they measured (~1.5e-2).

Identity check after the seed fix: layer-27 top-5 readouts are **identical**
to the logit lens on all ball_drop groups (J_27 = I, as the paper requires).

Gate 0 (NF4-runtime vs true-weight drift, the gap the averaged lens must
tolerate): 7–14% rel on single-layer branch deltas, cos ≥ 0.99
(`reports/jlens_gate0.json`) — recorded in the lens meta caveats.

## Grid, before/after (criterion 2) — ball_drop, "Why does the ball fall?"

Logit lens (before):

```
layer | g0 0-2s            | g1 2-4s            | g2 4-6s            | g3 6-8s
    0 | ·                  | ·                  | ·                  | okable,ascript
    8 | longleftrightarrow | registrazione,long | registrazione,IFn  | vla,MLElement
   12 | longleftrightarrow | registrazione,ummi | ummings,registrazi | registrazione,long
   16 | GuidId,longleftrig | GuidId             | GuidId,UsageId     | GuidId,vla
   24 | GuidId             | GuidId             | GuidId             | oproject,ibrator
   27 | gray,the           | red,brown          | the,brown          | the,brown
```

J-lens, identity seed (after):

```
layer | g0 0-2s            | g1 2-4s            | g2 4-6s            | g3 6-8s
    0 | the                | the                | the                | the
    8 | the,to             | the,to             | the,to             | the,to
   12 | the,to             | the,to             | the,to             | the,to
   16 | the,two            | the,two            | the                | the
   20 | two                | two                | two                | one
   24 | level              | two,level          | horizontal,move    | off,right
   27 | gray,the           | red,brown          | the,brown          | the,brown
```

Layer 24 is the headline: **temporally differentiated motion/relation
content** — `level` while the ball sits on the table, `horizontal, move`
while it rolls, `off, right` as it falls off the right edge — where the
logit lens reads `GuidId` and the norm-seeded variant read
`printStats/addCriterion` junk. These relation words never appear at layer
27 (which reads colors/objects), i.e. the band 22–26 exposes workspace-like
content the output layer itself does not.

Content-token onset (ball/brown/roll/move/off/level/horizontal/... anywhere
in top-15):

| lens | content onset | density at 24 (hits across 4 groups) |
|---|---|---|
| logit | layer 25 | 0 |
| j-lens norm-seed (repo convention) | 22 (weak) | 3 |
| j-lens identity seed (paper) | 22 | **9, temporally coherent** |

Answer-token classification (workspace_range.py): clean echo -> workspace ->
motor progression — echo 2–10% at layers 0–17, motor rising monotonically
8% (L0) -> 23% (L8) -> 38% (L20) -> **90% (L27, = logit lens exactly)**;
junk 2–30% vs logit's 34–82%. The norm-seeded variant's late-layer motor
regression (L27 90% -> 34%) is gone by construction.

Mid layers 8–20 on visual patches: still function words (`the`, `to`,
`two`) — patch content below layer ~22 remains not linearly readable under
a corpus-averaged J. Suggested UI display band (visual+answer combined
score): layers 6–13 for answer-workspace viewing; content band for patches
is 22–27.

**Verdict on criterion 2: PARTIAL PASS.** Mid layers 8–20 stay content-free
on patches (the honest negative survives), but the corrected lens unlocks a
real content band at 22–26 with motion/spatial relations absent from both
the logit lens and the output layer, plus a dramatically cleaner answer-token
workspace view at all layers.

## Concept recall (criterion 3) — PASS after the seed fix

Baseline stats refit per lens (`baseline_stats_jlens.json`), M2 gate re-run
(`reports/m2_quality_gate_jlens.json`), same 10 clips, layer floor 22:

|  | logit-lens-v1 | j-lens norm-seed | j-lens identity seed |
|---|---|---|---|
| concepts surfaced (all clips) | 26 | 19 | **34** |
| known-content elements covered | 12/54 | 11/54 | **16/54** |

Recall moves up +31% / +33% with the paper-faithful seed (it moved DOWN
under the repo's seed — the extra final-norm factor smeared the late band).
8 of 10 clips improve or hold; new concepts are content ("ball", "square",
"smaller", "inside", "purple", "background"). Concepts remain experimental
(precision re-judgment pending); raw grid stays primary per the M2 fallback.

## Timing (criterion 4) — PASS

`_timing_traffic_15s.mp4` with `--lens j-lens-v1`: **16.1 s total including
model load** (lens_decode 1.22 s — J application is one extra 3584x3584
fp16 matvec per readout). Budget: 90 s.

## Overall verdict

Verified port; the arbiter question splits cleanly:

- **Mid-layer (8–20) patch content: verified negative.** Even a correct,
  paper-faithful averaged J-lens reads only function words there.
  Per the PRD decision tree the answer-token x layer surface stays the hero.
- **Late-mid band (22–26): real unlock.** Temporally differentiated
  motion/relation readouts invisible to the logit lens; content onset moves
  25 -> 22; M2 concept recall +31%.
- **Answer-token view: unambiguous win** (echo->workspace->motor visible,
  junk collapsed, late layers exactly = logit lens).

`--lens j-lens-v1` stays opt-in for v0.2 (logit lens default) pending a
precision re-judgment of the new concepts; flipping the default is a
one-line change once judged.

Process note, recorded for honesty: the first fit ported the reference
repo's chain-seed convention, whose late-layer damage initially flipped
criteria 2/3 to negative. Checking the implementation against the PAPER's
formulation (not just the repo) is what caught it.
