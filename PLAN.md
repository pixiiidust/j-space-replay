# J-Space-Replay — Production Plan

Goal: ship a polished, self-hostable local app (open-source repo) that runs on
a single consumer GPU. "Production" = a stranger with an RTX-class 16 GB card
can clone, run one setup command, upload a clip, and get a replay — reliably,
with honest labeling.

Reference hardware: RTX 5070 Ti (16 GB, Windows 11). Everything below is
sequenced so each milestone produces something runnable.

---

## Prerequisites (verify before Milestone 0)

**Hardware**
- [ ] NVIDIA RTX 5070 Ti (16 GB VRAM). Minimum for others: 12 GB + tighter caps
- [ ] 32 GB system RAM recommended (activation capture streams to CPU), 16 GB minimum
- [ ] ~30 GB free disk: model weights (~6–7 GB), Python env incl. torch cu128 (~10 GB), HF cache, clips/traces

**Drivers / runtime**
- [ ] NVIDIA driver R570+ (Blackwell / CUDA 12.8 support) — check `nvidia-smi`
- [ ] No separate CUDA Toolkit install needed (PyTorch wheels bundle it)

**Software**
- [ ] Python 3.11 or 3.12 (not 3.13 — ML wheel lag) via `uv`
- [ ] Node 20+ and npm (frontend build)
- [ ] git; `hf` CLI for model download (Qwen2.5-VL-7B-Instruct is ungated — no token required)
- [ ] No C++ build tools required — we deliberately avoid anything that compiles on Windows (flash-attn, awq_ext, triton-dependent kernels)

**Quantization note (Windows-specific):** AutoAWQ's CUDA kernels have
unreliable Windows wheels. Default to **bitsandbytes NF4** (official Windows
wheels since 0.43, zero compilation): `BitsAndBytesConfig(load_in_4bit=True,
bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)`. Use AWQ/GPTQ
only if prebuilt wheels install cleanly. Activations are fp16 either way —
lens quality is unaffected by weight quantization format.

**Knowledge/assets**
- [ ] 3 short test clips (5–20 s) with known content for fixtures, e.g. person slipping, ball thrown, door opening
- [ ] Skim the workspace paper §J-lens and the jlens-qwen36 README before M1 — the position-map and readout code mirror their patterns

---

## Milestone 0 — Environment bring-up (½ day)

- [ ] Python env (uv), PyTorch cu128+ wheel, verify `torch.cuda.get_device_capability()` reports sm_120
- [ ] Download Qwen2.5-VL-7B-Instruct; load 4-bit via bitsandbytes NF4 (see prereqs), `attn_implementation="sdpa"`
- [ ] One text-only forward pass; one image pass; one 5 s video pass via qwen-vl-utils + PyAV/torchvision
- [ ] Record peak VRAM for the video pass

**Exit criteria:** video Q&A works end-to-end in a script; peak VRAM ≤ 12 GB
at 1 FPS / 480p / `max_pixels` cap, leaving headroom for capture buffers.

## Milestone 1 — Trace pipeline, CLI only (3–5 days)

The product's core. No UI yet: `python -m jsr.trace clip.mp4 "Why did the person fall?" -o trace.json`

- [ ] Frame sampler: 1 FPS default, `max_pixels` enforced, frame-group ↔ timestamp bookkeeping (2-frame temporal merge)
- [ ] Forward hooks on every decoder layer; residuals streamed to CPU inside the hook (assert no GPU accumulation)
- [ ] Position map: token index → {visual: frame group + patch (row, col)} | {text} | {generated}
- [ ] Logit-lens decode: `unembed(norm(residual))` top-k per (position, layer), fp16, batched on GPU then offloaded
- [ ] Answer generation with per-step capture at generated positions
- [ ] trace.json writer per SPEC schema (raw readouts + meta; concepts empty for now)
- [ ] Golden tests: 3 fixture clips with committed expected shapes (token counts, layer counts, timing map)

**Exit criteria:** a 15 s clip yields a valid trace in ≤ 90 s; the raw
frame-group × layer token grid, eyeballed, shows the expected early-echo →
mid-workspace → late-answer progression on at least one fixture clip.

## Milestone 2 — Label extraction + normalization (3–5 days, highest product risk)

This decides whether the UI shows "wet floor" or `▁the oor ▁of`.

- [ ] Stopword/fragment filter + subword merge over top-k readouts
- [ ] Baseline stats: run N≈20 varied clips, store per-layer readout logit mean/std → z-score normalization
- [ ] Candidate phrase vocabulary: model's own answer + a caption pass + a general concept list
- [ ] Cluster readouts across (frame group, layer) by co-occurrence + embedding similarity; match to candidates
- [ ] Emit concepts with `source_tokens` provenance; keep raw readouts alongside — UI must always drill down
- [ ] Quality gate: hand-label 10 clips; ≥ 6 of top-10 concepts per clip judged sensible by a human

**Exit criteria:** quality gate passes. If it doesn't, ship the raw-token grid
as the primary view (jlens-qwen36 does exactly this) and demote concept labels
to "experimental" — do not block the ship on this.

## Milestone 3 — Backend service (2–3 days)

- [ ] FastAPI: `POST /videos` (upload), `POST /traces` (video_id + question → job), `GET /jobs/{id}` (staged progress), `GET /traces/{id}`, `GET /library`
- [ ] Job queue: strictly one GPU job at a time; queued jobs report position
- [ ] Progress stages emitted by the pipeline: sampling → prefill+capture → generating → lens decode → labels → grounding → done (SSE or polling)
- [ ] Trace store on disk keyed by (video hash, question hash) — cache hit returns instantly
- [ ] Failure paths: OOM → auto-retry once at lower max_pixels/FPS with a warning; corrupt video → clear error
- [ ] Grounding queries (model-native boxes) for top ~5 concepts as a pipeline stage

**Exit criteria:** two traces requested concurrently queue correctly; kill -9
during a job leaves no corrupt state; re-request of a cached trace is instant.

## Milestone 4 — Frontend (5–8 days)

Per SPEC visual style (research control room, monospace, #113f8c navy).

- [ ] Upload → question form (default question prefilled) → job progress screen with pipeline stages → auto-transition to replay
- [ ] Video player synced to trace clock; scrubber snaps to frame groups
- [ ] Timeline heatmap (canvas): concepts × frame groups, hover/click inspect
- [ ] J-space slice viewer: frame-group × layer and answer-token × layer grids, cell → top-10 raw tokens
- [ ] Concept board with peaks, pin/hide; event log (derived events, measurement phrasing)
- [ ] Overlays: grounding boxes; patch heatmap for selected concept
- [ ] Library screen: past traces, re-open instantly; "ask another question" re-runs pipeline on same video
- [ ] Honesty banner (per SPEC) on the replay screen

**Exit criteria:** full loop — upload, wait with visible progress, replay,
scrub, drill into a cell, re-ask a question — with no dev-tools knowledge.

## Milestone 5 — Hardening + packaging + ship (3–5 days)

- [ ] Single-command start: `jsr up` launches backend + serves built frontend on localhost
- [ ] Windows-native install path (uv + npm build); Linux/WSL2 CI job to keep it portable
- [ ] Input limits enforced with friendly errors: ≤ 25 s, ≤ 100 MB, common codecs; reject/transcode the rest
- [ ] VRAM guard: pre-flight check, refuse with guidance below 12 GB free
- [ ] 3 bundled demo clips + pre-baked traces so the app demos instantly with no GPU wait
- [ ] README: what it is, what it is NOT (honesty section), GPU requirements, setup, architecture diagram
- [ ] License audit (Qwen2.5-VL Apache 2.0 ✓), model download script with checksum
- [ ] Version the trace schema (`"schema": 1`); viewer rejects unknown versions cleanly
- [ ] Tag v0.1.0, publish repo, demo video/GIF for the README

**Exit criteria (Definition of Shipped):** fresh Windows machine with a 16 GB
GPU goes from `git clone` to replaying a bundled demo in < 10 min, and to
replaying their own clip in < 20 min including model download.

## Post-ship (v0.2+)

- Tuned lens (per-layer affine translators) — cleaner readouts, same UI
- J-lens per the borrow plan below (revised estimate: **3–5 days**, was 1–2 weeks)
- Compare-two-questions view; concept pinning across traces
- Optional hosted demo (needs a GPU box + queue; out of scope for v0.1)

## J-lens borrow plan (from jlens-qwen36, Apache 2.0 — attribution required)

The reference repo's fitting recipe removes most of the Jacobian engineering
risk. Key facts extracted from its source (fit_analytic.py, analytic_layer.py):

**The recipe (port to PyTorch, don't port the MLX code):**
1. Per-layer Jacobians are computed **analytically from layer structure**, not
   autograd — ~30–60x faster than VJP fitting:
   - RMSNorm: closed-form, diag + rank-1 (`rms_norm_jacobian`)
   - SwiGLU MLP: "Hadamard trick" — element-wise fold of activation
     derivatives into weight matrices (`mlp_branch_jacobian`)
   - Softmax attention: batch identity cotangents in head space through the
     softmax core (`attn_branch_jacobian`)
2. Chain **top-down**: `J_{l-1} = J_l @ M_l`, one sweep from final norm to
   early layers gives the lens for every layer at once.
3. **Storage is D×D per layer, not D×vocab**: J_ℓ ∈ R^{3584×3584} ≈ 50 MB
   fp32 → ~1.4 GB for all 28 layers. Unembedding is applied at read time:
   `softmax(W_U · norm(J_ℓ · h))`. (Supersedes the earlier 1.1 GB/layer
   estimate — no vocab restriction needed for storage.)
4. Fit = average J across a prompt corpus by summation (`J_sum[l] += J`),
   with a stated position-averaging approximation at branch junctions.
5. Verify each analytic component against autograd on a **single layer**
   (`verify_analytic_layer.py` pattern — torch.func.jacrev on one layer is
   cheap and is exact ground truth).

**Why our port is EASIER than the original:** the repo's hard 80% — custom
Metal backward kernels for Qwen3.6's GatedDeltaNet linear-attention layers
(48 of 64 layers, ~27 s/layer, zeroed decay gates) — is irrelevant to us.
Qwen2.5-VL is 100% standard softmax attention, the layer type the repo
handles in ~1.4 s/layer with no custom kernels. Estimated fit time for
28 layers × 50–200 video prompts: well under an hour on the 5070 Ti.

**Also borrow:** `workspace_range.py` (finds which layers show workspace-like
readouts → picks our 6–8 display layers), the finite-difference verification
discipline, the single-file no-build-step web UI pattern, FastAPI serve
structure, and the honesty framing ("demo-quality, not research-grade").

**Doesn't transfer:** any MLX/Metal code, all GDN machinery, their prompt
corpus (text-only — ours must be video+question prompts). Our video position
map (M1) has no counterpart in their repo; that work stays ours.

**Port sequence (3–5 days):** RMSNorm Jacobian + single-layer autograd verify
→ SwiGLU Hadamard + verify → attention branch + verify → top-down chain →
fit over video corpus → drop into trace schema as `"lens": "j-lens-v1"`.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Label extraction quality too low | **High** | Milestone 2 gate: fall back to raw-token grid as primary view; labels marked experimental |
| Readouts on visual tokens are mush (method unvalidated on VLMs) | Medium | Two-stage gate: (1) M1 logit-lens eyeball check (~day 4: free 30-line decode on plumbing J-lens needs anyway; also = the identity first link of the J-lens chain) — structure means signal exists, proceed; static is NOT conclusive (logit lens fails on misaligned bases where fitted lenses succeed). (2) If static, go straight to the J-lens port (3–5 days per borrow plan) as the arbiter — tuned lens no longer needed as intermediate. Only if J-lens is also static: pivot to answer-token × layer as hero surface (text-token regime, method validated there) |
| VRAM overrun on long/high-FPS clips | Medium | Hard input caps + max_pixels + auto-retry at lower settings |
| Blackwell/Windows dependency breakage | Low-Med | Pin exact torch/transformers versions; CI on Linux to detect drift |
| Overclaiming (users read it as "model's thoughts") | Medium | Honesty banner, "readout strength" axis labels, measurement-phrased events |

## Timeline

Solo, focused: **~3–4 weeks to v0.1.0.** Critical path is M1 → M2; the
frontend (M4) can start against fixture traces as soon as M1's schema is
stable.
