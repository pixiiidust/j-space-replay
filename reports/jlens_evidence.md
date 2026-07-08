# J-lens port evidence (issue #8)

Run on Pixie (RTX 5070 Ti, torch 2.11.0+cu128, transformers 5.13, NF4+SDPA),
2026-07-08. Lens fit by `scripts/fit_jlens.py` over 15 video prompts
(3 fixtures + 7 gate clips + 5 baseline clips, default question, prefill
positions, skip_first=4); ~178 s/layer, ~80 min total on the 5070 Ti.

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

Gate 0 (NF4-runtime vs true-weight drift, the gap the averaged lens must
tolerate): 7–14% rel on single-layer branch deltas, cos ≥ 0.99
(`reports/jlens_gate0.json`) — recorded in the lens meta caveats.

## Mid-layer grid, before/after (criterion 2) — ball_drop, "Why does the ball fall?"

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

J-lens (after):

```
layer | g0 0-2s            | g1 2-4s            | g2 4-6s            | g3 6-8s
    0 | all                | all                | all                | all,test
    8 | all,the            | all,the            | all,the            | all,the
   12 | all,only           | all,only           | all,only           | all,only
   16 | all,more           | all,just           | all,just           | all,more
   24 | printStats,matchCo | GuidId,twor        | addCriterion,GuidI | longer,addCriterio
   27 | balance,sees       | red,gray           | brown,sees         | brown,the
```

Per-layer visual wordlike share (patch top-1): layers 6–15 move from 9–25%
(logit) to **36–54%** (j-lens); answer-position junk collapses from 50–82%
to 6–24% at every layer.

Content-token scan (ball/red/brown/fall/roll/balance/... anywhere in top-15):

- logit lens: content **only at layers 25–27**.
- j-lens: content from **layer 22** (`balance`, `ball`, `left` at 22–23,
  shares 0.03–0.08), then the same late-layer band.

**Verdict on criterion 2: verified partial negative.** The port is verified
(criterion 1), and the transport demonstrably fixes the mid-layer BASIS —
readouts become real tokens instead of `registrazione`/`GuidId` junk — but
what is linearly readable there under a corpus-averaged Jacobian is generic
function words (`all`, `the`, `only`, `more`), uniform across time. Wordlike:
yes. Temporally differentiated content at layers 8–20: **no**. The content
band extends down ~3 layers (25→22), not into the mid stack.

Answer-token side (bonus, not a criterion): the j-lens makes mid-layer answer
readouts dramatically more workspace-like (50–74% wordlike-non-echo-non-motor
vs 18–40%), with motor onset visible from layer ~3; but late-layer motor
fidelity REGRESSES (L27 motor 90% → 34%) because the averaged final-norm
factor smears the already-aligned top layers. The logit lens therefore stays
the default; `--lens j-lens-v1` is the opt-in.

Per the PRD decision tree (issue #1) and SPEC honesty framing: mid-layer
patch content stays unavailable after a verified port -> the hero surface
remains answer-token x layer (already the UI hero since PR #19), and this
negative is recorded rather than papered over.

## Concept recall (criterion 3) — verified negative

Baseline stats refit with the j-lens (`baseline_stats_jlens.json`), concept
layer floor set from the evidence above (22 for j-lens vs 20 for logit),
M2 gate re-run (`reports/m2_quality_gate_jlens.json`), same 10 clips:

|  | logit-lens-v1 | j-lens-v1 |
|---|---|---|
| concepts surfaced (all clips) | 26 | **19** |
| known-content elements covered | 12/54 | **11/54** |

Recall does NOT move up — it moves slightly down. The two causes are visible
in the grids: (1) the newly wordlike mid layers contain function words, not
content, so nothing new matches candidates; (2) the averaged transport
slightly smears the late-layer band where the logit lens was already
vocabulary-aligned, lowering content patch-shares (e.g. brown 0.11 -> 0.07
peaks on ball_drop). The M2 "concepts experimental" fallback stands.

## Timing (criterion 4) — pass

J application is one extra 3584x3584 fp16 matvec per readout.
`_timing_traffic_15s.mp4` with `--lens j-lens-v1`: **17.8 s total including
model load** (lens_decode 1.23 s vs 1.22 s logit at M1) — well inside 90 s.

## Overall verdict

Verified port, honest negative on the product question: the J-lens fixes the
mid-layer readout BASIS but does not surface mid-layer visual content or
improve concept recall on Qwen2.5-VL video prompts. Per the PRD decision
tree: the hero surface stays answer-token x layer (already the UI hero) with
the late-layer patch band; `--lens j-lens-v1` remains an opt-in research
lens whose genuine improvement is answer-token workspace readability at mid
layers (junk 50-82% -> 6-24%). Logit lens stays the default.
