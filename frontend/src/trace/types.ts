/** TypeScript view of the schema-v1 trace (see src/jsr/schema.py). */

export interface RawReadout {
  layer: number;
  top_tokens: string[];
  strengths: number[];
  /** per-patch top-1 token ids, row-major over patch_grid [rows, cols] */
  patch_top1?: number[];
}

export interface Concept {
  label: string;
  strength: number;
  layer: number;
  source_tokens: string[];
}

export interface FrameGroup {
  group: number;
  time_start: number;
  time_end: number;
  frame_indices: number[];
  n_patch_tokens: number;
  patch_grid: [number, number];
  raw_readouts: RawReadout[];
  concepts: Concept[];
}

export interface AnswerReadout {
  top_tokens: string[];
  strengths: number[];
}

export interface AnswerToken {
  token: string;
  /** keyed by layer number (as string) */
  readouts_by_layer: Record<string, AnswerReadout>;
}

export interface Grounding {
  label: string;
  /** [x1,y1,x2,y2] in original video pixels */
  box: [number, number, number, number];
  time: number;
  source: string;
}

export interface TraceMeta {
  model: string;
  lens: string;
  temporal_resolution_frames: number;
  strength_normalization: string;
  n_layers: number;
  /** id -> display string, for decoding patch_top1 */
  token_strings?: Record<string, string>;
  [k: string]: unknown;
}

export interface Trace {
  schema: number;
  video_id: string;
  question: string;
  answer: string;
  meta: TraceMeta;
  frame_groups: FrameGroup[];
  answer_tokens: AnswerToken[];
  grounding: Grounding[];
}

export interface LibraryItem {
  trace_id: string;
  video_id: string;
  question: string;
  answer: string;
  created_at: string;
  duration_s: number;
}
