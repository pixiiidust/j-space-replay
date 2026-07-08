# Video J-Space Replay — Product Spec (corrected)

## One-line product

Upload a short video, ask a question, then replay the model's decoded internal
concept readouts synced to the video timeline.

## Honesty banner (must appear in UI)

> Demo-quality interpretability. Lens readouts are noisy, single-token, and
> unvalidated on vision-language models. The J-lens method was validated on
> Claude text models only (Anthropic workspace paper); this tool extrapolates
> it to a VLM. Not suitable for mechanistic claims.

## References

- Research anchor: Anthropic, "Verbalizable Representations Form a Global
  Workspace in Language Models" (transformer-circuits.pub/2026/workspace).
  J-lens = average linearized effect of an activation on token likelihood;
  J-space = sparse subspace of verbalizable concepts (~10–25 active at once).
  **Validated on Claude text models over ~1,000 text prompts — not on VLMs.**
- Implementation pattern: `WeZZard/jlens-qwen36` (MLX / Apple Silicon).
  Do NOT port MLX. Known limitations that transfer to this project:
  noisy/underfit with small prompt sets, **single-token concepts only**.
- Model: Qwen2.5-VL-7B-Instruct (Apache 2.0). Video understanding with
  dynamic FPS sampling and absolute time encoding (M-RoPE).

## Target hardware (locked)

- GPU: RTX 5070 Ti, 16 GB VRAM (Blackwell, sm_120)
- OS: Windows 11
- Stack: PyTorch cu128+ wheel, HF `transformers` with forward hooks,
  `attn_implementation="sdpa"` (no flash-attn on Windows),
  4-bit weights via bitsandbytes NF4 (~5.5 GB; Windows-safe, no compilation;
  AWQ/GPTQ only if wheels install cleanly; activations remain fp16 either way),
  video decode via torchvision or PyAV (NOT decord — unmaintained, flaky on Windows).
- **No vLLM / serving frameworks** — they hide the activations we need, and
  vLLM doesn't run natively on Windows anyway.

## VRAM / token budget (the real constraint)

- Weights 4-bit: ~5.5 GB. Remaining ~10 GB for KV cache + capture buffers.
- Qwen2.5-VL: 14 px patches, 2×2 spatial merge, 2-frame temporal merge
  → ~400–500 tokens per 2-frame group at 480p.
- Cap with processor `max_pixels` to ~250 tokens per frame group.
- Defaults: 1 FPS, 480p, 5–20 s clips → ≤ ~3k visual tokens per pass.
- Activation capture: stream each layer's hidden states to CPU **inside the
  hook**; never accumulate on GPU (~1 GB per pass at 5k tokens × 28 layers).

---

# Core user story

As a user, I upload a short video clip and ask a question (or accept the
default question "Describe what happens in this video").
The system runs one offline processing pass — sample frames, forward pass with
activation capture, lens decode, label extraction — showing pipeline progress.
When the trace is ready (~1–3 min for a 15 s clip), the replay view loads
automatically: I scrub the video and watch concept readouts strengthen, peak,
and fade over time.

**A trace is per (video, question) pair.** Asking a new question about the
same video is a new processing run. Traces are cached and re-openable
instantly.

---

# MVP input

- Video: 5–20 seconds
- Resolution: 480p default (enforced via `max_pixels`)
- Sampling: 1 FPS default, up to 4 FPS (warn about token budget)
- Question: free text, with default fallback
- Model: local Qwen2.5-VL-7B-Instruct, 4-bit
- Output: trace file, replayed in UI

# MVP output — trace format (corrected)

What the lens actually emits is top-k **vocabulary tokens** per (position,
layer) — often fragments (`▁wet`, `▁floor`, `oor`) — not phrases. Concept
labels are a **derived** layer produced by the label-extraction stage. Scores
are **normalized readout strength**, not probabilities. Boxes come from a
separate grounding query to the model, not from activations.

```json
{
  "video_id": "clip_001",
  "question": "Why did the person fall?",
  "answer": "The person likely slipped on the wet floor.",
  "meta": {
    "model": "qwen2.5-vl-7b-awq",
    "lens": "logit-lens-v1",
    "temporal_resolution_frames": 2,
    "strength_normalization": "per-layer z-score vs baseline corpus"
  },
  "frame_groups": [
    {
      "time_start": 4.0, "time_end": 6.0, "frame_indices": [8, 9],
      "raw_readouts": [
        { "layer": 24, "top_tokens": ["▁wet", "▁floor", "▁slip"], "strengths": [2.9, 2.4, 2.1] }
      ],
      "concepts": [
        { "label": "wet floor", "strength": 2.7, "layer": 24, "source_tokens": ["▁wet", "▁floor"] }
      ],
      "patch_heatmap": { "grid": [15, 10], "concept": "wet floor", "values": "…" }
    }
  ],
  "answer_tokens": [
    { "token": "▁slipped", "readouts_by_layer": { "20": ["▁fell", "▁slip"], "28": ["▁slipped"] } }
  ],
  "grounding": [
    { "label": "floor", "box": [120, 300, 480, 420], "time": 4.5, "source": "model-grounding-query" }
  ]
}
```

---

# What the timeline measures (framing, locked)

- **frame × layer view**: readouts at visual-token positions during prefill =
  "what representations exist at each frame group's tokens, per layer."
  Causal attention means later frame groups legitimately accumulate context.
  Finest time resolution is a **2-frame group**, not a single frame.
- **answer-token × layer view**: the answer forming during generation.
- **Event log**: derived **mechanically only** — concept crosses threshold /
  peaks / drops. Phrase events as measurements ("wet-floor readout crosses
  z=2.0 at layer 24, t=4.5s"), never as narrative ("model realizes…").
- **Regions**: activation-derived localization is a coarse patch heatmap
  (~15×10). Clean bounding boxes come from Qwen2.5-VL's native grounding
  (ask the model for coordinates) as a separate overlay channel.
  Hybrid: lens for concepts-over-time, grounding for where.

---

# Label extraction (its own subsystem — budget real time)

Token readouts → concept labels is the difference between shipping the mock
and shipping `▁the ▁floor oor ▁of` in a grid.

Pipeline:
1. Collect top-k readout tokens per (frame group, layer).
2. Filter stopwords/fragments; merge subword pieces.
3. Cluster across frames/layers (co-occurrence + embedding similarity).
4. Match clusters against a candidate phrase vocabulary (built from the
   model's own answer + caption pass + a general concept list).
5. Emit concept labels with provenance (`source_tokens`) so the UI can always
   drop to raw readouts on click.

Normalization: per-layer z-score of readout logit vs a baseline corpus of
clips (final calibration method may be revised after Phase 2 measurement).
UI axis label: **"readout strength"** — never "confidence" or "probability."

---

# Interface spec

## Visual style

- light background, thin grey borders, navy active states, tiny grid cells,
  monospaced labels, lots of whitespace, technical replay dashboard feel
- no glossy UI, no gradients, no rounded SaaS cards
- "research control room," not "consumer app"

Palette:

```txt
background: #f7f7f5   panel: #ffffff    border: #d8d8d2   grid: #e8e8e2
primary: #113f8c      text: #1d1d1b     muted: #8b8b84    warning: #9b2f5f
```

Typography: IBM Plex Mono / JetBrains Mono; titles uppercase letter-spaced;
body small, dense; numbers tabular.

## Main screen layout

```txt
VIDEO J-SPACE REPLAY
clip 12s | frame groups 6 | layers 28 | concepts 42 | model qwen2.5-vl-7b | lens logit-v1

[ dashboard ] [ replay ] [ lens ] [ events ]

LEFT PANEL                 CENTER CANVAS                RIGHT PANEL
VIDEO PLAYER               J-SPACE TIMELINE             CONCEPTS
current frame              frame-group × concept grid   top active readouts
0.25x 0.5x 1x              layer bands                  pinned concepts
scrubber                   readout strength heat        event log (derived)

FRAME REGIONS              WORKSPACE SLICE              DETAILS
grounding boxes            layer × position grid        selected concept
patch heatmap              token readout table          layer trajectory
                                                        raw-token drill-down
```

## Playback behavior

Controls: play/pause · 0.25x/0.5x/0.75x/1x · step frame group ·
jump to concept peak · pin/hide concept · compare layers.

During playback:

```txt
00:04.50
active (readout strength, z):
wet floor      ████████ 2.7
foot contact   ██████   2.1
slip           █████    1.9
```

## Panels

1. **Video panel** — clean / grounding boxes / patch-heatmap overlay.
2. **Timeline heatmap** — main surface. Rows: concepts (and optionally layers);
   columns: frame groups. Cell = readout strength; click to inspect; hover
   shows frame group, layer, concept, strength, source tokens.
3. **J-space slice viewer** — frame-group × layer; answer-token × layer;
   pinned-concept × time. Every cell drills down to raw top-10 tokens.
4. **Concept board** — derived concepts with peak times + provenance.
5. **Event log** — mechanically derived threshold/peak events only.

---

# Processing pipeline

```txt
upload video → sample frames (1 FPS, max_pixels cap)
→ build prompt (frames + question)
→ single forward pass, hooks stream residuals per layer to CPU
→ generation with per-token capture (answer)
→ logit-lens decode (unembed @ intermediate residuals)
→ label extraction + normalization
→ grounding queries for key concepts (separate small passes)
→ derive events → write trace → replay in UI
```

Process first, replay second. Never compute live during playback.

---

# Technical architecture

```txt
frontend   React/Next · video player · timeline canvas · grid renderer · inspector
backend    FastAPI · upload/ingest · job queue (one GPU job at a time) · trace store
model svc  PyTorch CUDA · transformers + hooks · logit-lens decoder · label extractor
storage    video file · sampled frames · trace.json · baseline stats
```

---

# Model plan (corrected phasing — fake-lens phase dropped)

The logit lens IS the cheap real thing (~30 lines of hook code, zero fitting).
Simulating a fake concept timeline is more work than computing a crude real
one, and fake data hides the actual UX problem: noise.

- **Phase 1 — logit lens**: unembedding applied to intermediate residuals at
  visual-token and answer-token positions. Plus label extraction v1.
- **Phase 2 — J-lens**: analytic per-layer Jacobians chained top-down per the
  jlens-qwen36 recipe (see PLAN.md borrow plan; 3–5 days — Qwen2.5-VL is all
  standard softmax attention, so no custom kernels); mark "demo-quality"
  until validated. Note: the logit lens is the identity first link of this
  chain — Phase 1 is contained in Phase 2, not thrown away.
- **(Optional) tuned lens**: per-layer affine translators — only as a
  contingency if the analytic J-lens port hits trouble.
- **Phase 4 — research-grade (post-ship)**: 100+ prompt lens fitting,
  full-depth layers, ablation, concept patching, counterfactual replay.

# MVP non-goals

- hour-long video · real-time analysis · video generation interpretability
- consciousness claims · perfect mechanistic proof
- 27B model on 16 GB · full Anthropic reproduction
- clean bounding boxes from activations (grounding queries only)
- calibrated probabilities (readout strength only)

# Name

**J-Space-Replay**
