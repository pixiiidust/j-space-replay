/**
 * Pure selectors over a trace: the trace clock (video time <-> frame group),
 * scrubber snapping, group stepping, and the timeline heatmap model
 * (concepts x frame-groups, with a wordlike-token fallback when a trace carries
 * no derived concept labels). All deterministic and unit-tested.
 */
import type { FrameGroup, Trace } from "./types";
import { isWordlike, normalizeToken, wordKey } from "./wordlike";

// ---------------------------------------------------------------- trace clock

/** Index of the frame group covering video time `t` (clamped to [0, n-1]). */
export function groupIndexAtTime(trace: Trace, t: number): number {
  const groups = trace.frame_groups;
  if (groups.length === 0) return -1;
  if (t < groups[0].time_start) return 0;
  for (let i = 0; i < groups.length; i++) {
    const g = groups[i];
    // half-open [start, end); last group is closed so the final frame counts
    if (t >= g.time_start && (t < g.time_end || i === groups.length - 1)) {
      return i;
    }
  }
  return groups.length - 1;
}

/** Snap an arbitrary video time to the start of its containing frame group. */
export function snapToGroupStart(trace: Trace, t: number): number {
  const idx = groupIndexAtTime(trace, t);
  return idx < 0 ? 0 : trace.frame_groups[idx].time_start;
}

/** Move `idx` by `dir` (-1 / +1) frame groups, clamped to valid range. */
export function stepGroupIndex(trace: Trace, idx: number, dir: number): number {
  const n = trace.frame_groups.length;
  if (n === 0) return -1;
  return Math.max(0, Math.min(n - 1, idx + Math.sign(dir)));
}

/** Total clip duration = end of the last frame group. */
export function traceDuration(trace: Trace): number {
  const g = trace.frame_groups;
  return g.length ? g[g.length - 1].time_end : 0;
}

// ------------------------------------------------------------- timeline model

export type TimelineMode = "concepts" | "fallback";

export interface TimelineRow {
  label: string;
  kind: "concept" | "token";
  sourceTokens: string[];
  /** strength per frame group, null where the row is absent that group */
  cells: (number | null)[];
  /** dominant layer per frame group (null where absent) */
  layers: (number | null)[];
  peakGroup: number;
  peakStrength: number;
  peakLayer: number | null;
  peakTime: number;
}

export interface TimelineModel {
  mode: TimelineMode;
  rows: TimelineRow[];
  groups: FrameGroup[];
  maxStrength: number;
}

function finishRow(
  label: string,
  kind: "concept" | "token",
  sourceTokens: string[],
  cells: (number | null)[],
  layers: (number | null)[],
  groups: FrameGroup[],
): TimelineRow {
  let peakGroup = 0;
  let peakStrength = -Infinity;
  for (let i = 0; i < cells.length; i++) {
    const v = cells[i];
    if (v != null && v > peakStrength) {
      peakStrength = v;
      peakGroup = i;
    }
  }
  if (!isFinite(peakStrength)) peakStrength = 0;
  return {
    label,
    kind,
    sourceTokens,
    cells,
    layers,
    peakGroup,
    peakStrength,
    peakLayer: layers[peakGroup] ?? null,
    peakTime: groups[peakGroup]?.time_start ?? 0,
  };
}

/**
 * Concepts-mode timeline: one row per distinct concept label, cells are the
 * concept's strength in each frame group (max if a label repeats in a group).
 */
function buildConceptRows(groups: FrameGroup[]): TimelineRow[] {
  const order: string[] = [];
  const cellsByLabel = new Map<string, (number | null)[]>();
  const layersByLabel = new Map<string, (number | null)[]>();
  const srcByLabel = new Map<string, string[]>();

  for (let gi = 0; gi < groups.length; gi++) {
    for (const c of groups[gi].concepts) {
      if (!cellsByLabel.has(c.label)) {
        order.push(c.label);
        cellsByLabel.set(c.label, new Array(groups.length).fill(null));
        layersByLabel.set(c.label, new Array(groups.length).fill(null));
        srcByLabel.set(c.label, c.source_tokens ?? []);
      }
      const cells = cellsByLabel.get(c.label)!;
      const prev = cells[gi];
      if (prev == null || c.strength > prev) {
        cells[gi] = c.strength;
        layersByLabel.get(c.label)![gi] = c.layer;
      }
    }
  }

  return order
    .map((label) =>
      finishRow(
        label,
        "concept",
        srcByLabel.get(label)!,
        cellsByLabel.get(label)!,
        layersByLabel.get(label)!,
        groups,
      ),
    )
    .sort((a, b) => b.peakStrength - a.peakStrength);
}

/**
 * Fallback timeline derived from raw readouts: aggregate wordlike tokens per
 * frame group (max patch-share across layers), rank by peak strength, keep the
 * top `topN`.
 */
function buildFallbackRows(groups: FrameGroup[], topN: number): TimelineRow[] {
  const display = new Map<string, string>();
  const cells = new Map<string, (number | null)[]>();
  const layers = new Map<string, (number | null)[]>();

  const ensure = (key: string, disp: string) => {
    if (!cells.has(key)) {
      display.set(key, disp);
      cells.set(key, new Array(groups.length).fill(null));
      layers.set(key, new Array(groups.length).fill(null));
    }
  };

  for (let gi = 0; gi < groups.length; gi++) {
    for (const r of groups[gi].raw_readouts) {
      for (let k = 0; k < r.top_tokens.length; k++) {
        const tok = r.top_tokens[k];
        if (!isWordlike(tok)) continue;
        const key = wordKey(tok);
        ensure(key, normalizeToken(tok));
        const s = r.strengths[k] ?? 0;
        const prev = cells.get(key)![gi];
        if (prev == null || s > prev) {
          cells.get(key)![gi] = s;
          layers.get(key)![gi] = r.layer;
        }
      }
    }
  }

  const rows = [...cells.keys()].map((key) =>
    finishRow(display.get(key)!, "token", [display.get(key)!], cells.get(key)!, layers.get(key)!, groups),
  );
  rows.sort((a, b) => b.peakStrength - a.peakStrength);
  return rows.slice(0, topN);
}

export function buildTimeline(trace: Trace, topN = 14): TimelineModel {
  const groups = trace.frame_groups;
  const hasConcepts = groups.some((g) => g.concepts && g.concepts.length > 0);
  const rows = hasConcepts
    ? buildConceptRows(groups)
    : buildFallbackRows(groups, topN);
  let maxStrength = 0;
  for (const row of rows) {
    for (const c of row.cells) if (c != null && c > maxStrength) maxStrength = c;
  }
  return {
    mode: hasConcepts ? "concepts" : "fallback",
    rows,
    groups,
    maxStrength: maxStrength || 1,
  };
}

/** Active rows at a given frame group, strongest first (for the playback HUD). */
export function activeRowsAtGroup(
  model: TimelineModel,
  groupIdx: number,
  limit = 6,
): Array<{ label: string; strength: number }> {
  return model.rows
    .map((r) => ({ label: r.label, strength: r.cells[groupIdx] ?? 0 }))
    .filter((r) => r.strength > 0)
    .sort((a, b) => b.strength - a.strength)
    .slice(0, limit);
}
