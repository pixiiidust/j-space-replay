/**
 * J-space slice selectors: the frame-group x layer grid, the answer-token x
 * layer grid, per-cell raw top-k token drill-down, and the per-concept patch
 * heatmap derived from patch_top1 + meta.token_strings.
 */
import type { AnswerToken, FrameGroup, Trace } from "./types";
import { wordKey } from "./wordlike";

export interface TokenStrength {
  token: string;
  strength: number;
}

/** Top-k (token, strength) pairs from a readout-like {top_tokens, strengths}. */
export function topTokens(
  r: { top_tokens: string[]; strengths: number[] } | undefined,
  k = 10,
): TokenStrength[] {
  if (!r) return [];
  const out: TokenStrength[] = [];
  for (let i = 0; i < Math.min(k, r.top_tokens.length); i++) {
    out.push({ token: r.top_tokens[i], strength: r.strengths[i] ?? 0 });
  }
  return out;
}

/** Sorted list of layer numbers present in the frame-group readouts. */
export function groupLayers(trace: Trace): number[] {
  const set = new Set<number>();
  for (const g of trace.frame_groups) {
    for (const r of g.raw_readouts) set.add(r.layer);
  }
  return [...set].sort((a, b) => a - b);
}

/** frame-group x layer matrix; value = top (max) strength of that readout. */
export function groupLayerGrid(trace: Trace): {
  layers: number[];
  values: (number | null)[][]; // [groupIdx][layerIdx]
} {
  const layers = groupLayers(trace);
  const layerIndex = new Map(layers.map((l, i) => [l, i]));
  const values: (number | null)[][] = trace.frame_groups.map(() =>
    new Array(layers.length).fill(null),
  );
  trace.frame_groups.forEach((g, gi) => {
    for (const r of g.raw_readouts) {
      const li = layerIndex.get(r.layer);
      if (li == null) continue;
      values[gi][li] = r.strengths.length ? r.strengths[0] : null;
    }
  });
  return { layers, values };
}

export function rawTokensAt(
  group: FrameGroup | undefined,
  layer: number,
  k = 10,
): TokenStrength[] {
  const r = group?.raw_readouts.find((x) => x.layer === layer);
  return topTokens(r, k);
}

/** Sorted union of layer numbers present across the answer tokens. */
export function answerLayers(trace: Trace): number[] {
  const set = new Set<number>();
  for (const at of trace.answer_tokens) {
    for (const l of Object.keys(at.readouts_by_layer)) set.add(Number(l));
  }
  return [...set].sort((a, b) => a - b);
}

/** answer-token x layer matrix; value = top strength (raw logit) per cell. */
export function answerLayerGrid(trace: Trace): {
  layers: number[];
  values: (number | null)[][]; // [tokenIdx][layerIdx]
} {
  const layers = answerLayers(trace);
  const layerIndex = new Map(layers.map((l, i) => [l, i]));
  const values = trace.answer_tokens.map((at) => {
    const row: (number | null)[] = new Array(layers.length).fill(null);
    for (const [l, r] of Object.entries(at.readouts_by_layer)) {
      const li = layerIndex.get(Number(l));
      if (li == null) continue;
      row[li] = r.strengths.length ? r.strengths[0] : null;
    }
    return row;
  });
  return { layers, values };
}

export function answerTokensAt(
  at: AnswerToken | undefined,
  layer: number,
  k = 10,
): TokenStrength[] {
  return topTokens(at?.readouts_by_layer[String(layer)], k);
}

/**
 * Patch heatmap for a target word within one frame group: for each patch,
 * 1 where its top-1 token (decoded via token_strings) matches the target
 * word (by wordKey), else 0. Row-major over patch_grid [rows, cols].
 * Returns null if patch_top1 / token_strings unavailable.
 */
export function patchHeatmapForWord(
  group: FrameGroup | undefined,
  layer: number,
  tokenStrings: Record<string, string> | undefined,
  targetWord: string,
): { rows: number; cols: number; values: number[] } | null {
  if (!group || !tokenStrings) return null;
  const r = group.raw_readouts.find((x) => x.layer === layer);
  if (!r || !r.patch_top1) return null;
  const [rows, cols] = group.patch_grid;
  const target = wordKey(targetWord);
  const values = r.patch_top1.map((id) => {
    const s = tokenStrings[String(id)];
    if (s == null) return 0;
    return wordKey(s) === target ? 1 : 0;
  });
  return { rows, cols, values };
}

/**
 * Choose the layer whose readout most represents `targetWord` in a group
 * (the layer where that word has the highest strength). Falls back to a
 * mid-network layer when the word isn't found.
 */
export function bestLayerForWord(
  group: FrameGroup | undefined,
  targetWord: string,
): number | null {
  if (!group) return null;
  const target = wordKey(targetWord);
  let best: number | null = null;
  let bestStrength = -Infinity;
  for (const r of group.raw_readouts) {
    for (let i = 0; i < r.top_tokens.length; i++) {
      if (wordKey(r.top_tokens[i]) === target) {
        const s = r.strengths[i] ?? 0;
        if (s > bestStrength) {
          bestStrength = s;
          best = r.layer;
        }
      }
    }
  }
  if (best != null) return best;
  const mid = group.raw_readouts[Math.floor(group.raw_readouts.length / 2)];
  return mid ? mid.layer : null;
}
