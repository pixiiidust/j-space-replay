/** Small synthetic trace builders for deterministic unit tests. */
import type { FrameGroup, Trace } from "./types";

export function group(
  idx: number,
  overrides: Partial<FrameGroup> = {},
): FrameGroup {
  return {
    group: idx,
    time_start: idx * 2,
    time_end: idx * 2 + 2,
    frame_indices: [idx * 2, idx * 2 + 1],
    n_patch_tokens: 4,
    patch_grid: [2, 2],
    raw_readouts: [],
    concepts: [],
    ...overrides,
  };
}

export function makeTrace(groups: FrameGroup[], overrides: Partial<Trace> = {}): Trace {
  return {
    schema: 1,
    video_id: "test",
    question: "Describe what happens in this video.",
    answer: "an answer",
    meta: {
      model: "test",
      lens: "logit-lens-v1",
      temporal_resolution_frames: 2,
      strength_normalization: "test",
      n_layers: 4,
    },
    frame_groups: groups,
    answer_tokens: [],
    grounding: [],
    ...overrides,
  };
}
