/**
 * The honesty banner text — LOCKED wording (guarded verbatim by app.test.tsx).
 * Must render unchanged on the replay screen. Do not paraphrase.
 */
export const HONESTY_BANNER =
  "Demo-quality interpretability. Lens readouts are noisy, single-token, and " +
  "unvalidated on vision-language models. The J-lens method was validated on " +
  "Claude text models only (Anthropic workspace paper); this tool extrapolates " +
  "it to a VLM. Not suitable for mechanistic claims.";

export const DEFAULT_QUESTION = "Describe what happens in this video.";

/** Pipeline stages, in order, as reported by GET /jobs/{id}. */
export const PIPELINE_STAGES = [
  "sampling",
  "prefill_capture",
  "generating",
  "lens_decode",
  "labels",
  "grounding",
  "done",
] as const;

export const PLAYBACK_SPEEDS = [0.25, 0.5, 0.75, 1] as const;

/** Axis label for strength — SPEC locks this; never "confidence"/"probability". */
export const STRENGTH_AXIS_LABEL = "readout strength";

/**
 * Display names for the decode lenses. "logit lens" names the identity
 * readout METHOD (unembed(norm(h))); the strengths shown are raw unembedding
 * scores under either lens — keep the two ideas from blurring in UI copy.
 */
export const LENS_LABELS: Record<string, string> = {
  "logit-lens-v1": "logit lens",
  "j-lens-v1": "J-lens",
};
