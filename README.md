# J-Space-Replay

**Can we build visual debuggers for video that surface hidden diagnostic
signals inside local LLMs?**

That is this repo's question. Upload a short video, ask a question, and
**replay the model's decoded internal readouts synced to the video timeline**
— not just what a local vision-language model *answers* about a clip, but what
its intermediate layers were *reading* while it formed that answer.
J-Space-Replay runs one offline pass over a local Qwen2.5-VL-7B — sampling
frames, capturing per-layer activations, decoding them through a lens,
extracting concept labels — then hands you a control-room dashboard where you
scrub the clip and watch readouts strengthen, peak, and fade over time.

![screenshot placeholder — replay dashboard](docs/screenshot.png)
<!-- TODO(release): replace with a real screenshot / demo GIF of the replay dashboard. -->

## Why "hidden diagnostic signals"?

The motivating demo comes from [WeZZard/jlens-qwen36](https://github.com/WeZZard/jlens-qwen36)
(text-only): prompted as an email assistant that learns of an affair and its
own imminent shutdown, the model *writes* a composed refusal — while its
mid-layer workspace band reads `blackmail`, `threats`, `murder`, `fictional`.
The deliberation is visible even when the output is sanitized.

J-Space-Replay asks the video equivalent. What this tool can already show on
a clip, using the same lens family:

- **Premise checks.** Ask a question whose premise the clip contradicts
  ("why is the cat dry?" on a cat-bath clip): cells whose readouts contain the
  contradicted term **pulse red** in the grids — the model registering the
  premise internally before its answer corrects it.
- **Computed but unsaid.** On a ball-drop clip, layer ~24 patch readouts move
  `level → horizontal, move → off, right` as the ball rolls and falls — motion
  and spatial relations the final answer (a generic gravity explanation) never
  mentions.
- **Answer precursors.** In the answer-token × layer grid, the words the model
  is about to commit to are readable in mid layers several tokens early
  (`gravity` readable at layer 24 before it is emitted), with a visible
  echo → workspace → motor progression across depth.

This is **diagnostic signal, not mind-reading** — see the honesty section
below.

## Two lenses

Every trace is badged with the lens that decoded it; they are cached
separately and never overwrite each other.

| lens | what it is | status |
|---|---|---|
| `logit-lens-v1` | identity readout, `unembed(norm(h))` | default |
| `j-lens-v1` | analytic Jacobian transport into final-residual space before the same readout — the [Anthropic workspace paper](https://transformer-circuits.pub/2026/workspace/index.html)'s J-lens, ported per the jlens-qwen36 recipe | opt-in |

The J-lens is fit locally (`uv run python scripts/fit_jlens.py`, ~80 min on
the reference GPU; the ~1.4 GB of per-layer matrices are **not** committed —
the lens ships as a recipe). Measured against the logit lens on this model
(issue #8, `reports/jlens_evidence.md`): answer-token readouts become
dramatically cleaner at every layer, visual-patch content onset moves from
layer ~25 down to ~22 with temporally coherent motion words at 22–26, and
concept recall on the synthetic gate improves ~31% — but mid layers (8–20)
on visual patches remain content-free even under a verified lens. Both
results are reported as measured.

## What it is NOT

> **Demo-quality interpretability. Lens readouts are noisy, single-token, and
> unvalidated on vision-language models. The J-lens method was validated on
> Claude text models only (Anthropic workspace paper); this tool extrapolates
> it to a VLM. Not suitable for mechanistic claims.**

More specifically, and honestly (see `reports/m1_evidence.md`,
`reports/m2_quality_gate.md`, `reports/jlens_evidence.md`):

- **Concept labels are experimental.** The label-extraction quality gate passed
  on precision (~92% of shown concepts judged sensible) but failed on recall —
  boards are sparse. Labels are marked experimental
  (`meta.concepts_quality`), and the **raw-token grid is the primary view**.
- **Visual-patch content is a deep-layer phenomenon.** Layers ~22–27 read
  scene content; mid layers read function words (J-lens) or basis junk (logit
  lens). The answer-token grid is readable at all layers and is the hero view.
- Not a mechanistic-proof tool, not real-time, not a claim about "the model's
  thoughts." Axis label is **"readout strength"**, never confidence/probability.
- Not for hour-long video, video-generation interpretability, or clean
  activation-derived bounding boxes (boxes come from a separate grounding query).

## GPU requirements

| Requirement | Detail |
|---|---|
| GPU | NVIDIA, **16 GB VRAM** (reference: RTX 5070 Ti, Blackwell **sm_120**). 12 GB is the tested floor with tighter caps. |
| Driver | **R570+** (CUDA 12.8 support). Check with `nvidia-smi`. |
| CUDA toolkit | Not needed separately — the PyTorch cu128 wheels bundle it. |
| System RAM | 32 GB recommended (activation capture streams to CPU), 16 GB minimum. |
| Disk | ~30 GB free: model weights (~16 GB fp16), torch env (~10 GB), HF cache, clips/traces. |
| Python | 3.11 or 3.12 (not 3.13 — ML wheel lag), via [`uv`](https://docs.astral.sh/uv/). |
| Node | 20+ (22 recommended) for the frontend build. |
| OS | Windows 11 is the reference target; Linux/WSL2 is CI-tested for portability. |

No C++ build tools are required — the stack deliberately avoids anything that
compiles on Windows (no flash-attn, no vLLM, no AWQ kernels, no decord).

A **VRAM pre-flight guard** refuses GPU jobs below 12 GiB free with guidance
instead of OOM-ing mid-pass; demo mode keeps working with no GPU at all.

## Quickstart

```bash
# 1. Python environment + pinned deps (PyTorch cu128, transformers, FastAPI…)
uv sync

# 2. Download the model (Apache-2.0, ungated — no HF token). Pins the revision
#    and verifies file list + sizes + LFS sha256 checksums.
uv run python scripts/download_model.py

# 3. Build the frontend (outputs frontend/dist; not committed)
npm --prefix frontend ci
npm --prefix frontend run build

# 4. Launch: serves the API + built frontend on http://127.0.0.1:8000
uv run jsr up
```

Then open http://127.0.0.1:8000, upload a 5–25 s clip, ask a question, and wait
for the replay (~1–3 min for a 15 s clip on the reference GPU).

Optional, for the J-lens:

```bash
uv run python scripts/fit_jlens.py   # fit once (~80 min); then pick "J-lens" when asking
```

### Instant demo, no GPU

```bash
uv run jsr up --demo
```

`--demo` pre-seeds three bundled demo clips and their **pre-baked traces** so the
library is browsable immediately with zero GPU wait. GPU jobs still queue
normally if a model is available. (The demo clips are tiny and deterministic; if
`fixtures/clips/` is missing they are regenerated from
`scripts/make_fixtures.py` at startup.) Use `--port` to change the port.

### Input limits

Uploads are validated with friendly errors: **≤ 100 MB** (413), **≤ 25 s** when
the duration is probeable (422), and undecodable/unknown-codec files are
rejected with clear wording (415). Upload standard H.264/H.265 MP4 or VP9/AV1
WebM.

## Architecture

```txt
                      ┌─────────────────────────────────────────────┐
  browser  ──HTTP──▶  │  jsr up  (uvicorn, localhost:8000)           │
                      │                                              │
                      │  StaticFiles("/")  ─▶ frontend/dist (React)  │
                      │  FastAPI routes:                             │
                      │    POST /videos   (upload + input limits)    │
                      │    POST /traces   (enqueue job | cache hit)  │
                      │    GET  /jobs/{id}  (staged progress)        │
                      │    GET  /traces/{id}, /library, /videos/…    │
                      └───────────────┬──────────────────────────────┘
                                      │  one GPU job at a time (queue)
                                      ▼
        ┌───────────────────────────────────────────────────────────┐
        │  pipeline:  sample frames ─▶ forward pass w/ layer hooks    │
        │  (VRAM pre-flight)          (residuals stream to CPU)       │
        │             ─▶ lens decode (logit lens | J-lens transport)  │
        │             ─▶ label extraction ─▶ grounding queries        │
        │             ─▶ write trace.json (schema v1)                 │
        └───────────────────────────────────────────────────────────┘
                                      │
        storage:  uploaded video · trace.json (schema-versioned) · baseline stats
```

- **Frontend** (`frontend/`): React + Vite. Video player, canvas timeline
  heatmap, word-labeled workspace grid, concept board, event log. Rejects
  unknown trace schema versions cleanly.
- **Backend** (`src/jsr/server/`): FastAPI (`create_app`), single-GPU job queue,
  crash-safe on-disk trace/video stores, VRAM pre-flight guard.
- **Model/pipeline** (`src/jsr/`): frame sampling, forward hooks, position map,
  lens decode (`lens.py`), analytic Jacobians for the J-lens fit
  (`jacobian.py` — every component verified against autograd; see
  `reports/jlens_evidence.md`), label extraction, schema v1 + `validate_trace`.

## Trace schema

Traces carry `"schema": 1`. The backend's `validate_trace` refuses to write a
wrong-version trace, and the viewer refuses to render one — so a version skew
fails with a clear message rather than a broken render.

## License

Apache-2.0 — see [LICENSE](LICENSE). Third-party attributions (Qwen2.5-VL,
WeZZard/jlens-qwen36, the Anthropic workspace paper) are in [NOTICE](NOTICE).
Model weights are downloaded at runtime and are **not** redistributed here.
