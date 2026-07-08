/**
 * Event log — derived MECHANICALLY ONLY from the timeline strength series.
 * Three event kinds: a readout crossing a strength threshold (rising edge),
 * peaking (local maximum), and dropping back below the threshold.
 *
 * Phrasing is measurement-only, per SPEC §"What the timeline measures":
 *   "<label> readout crosses strength 0.50 at layer 24, t=4.5s"
 * NEVER narrative ("the model realizes ..."), NEVER "confidence"/"probability".
 */
import type { TimelineModel, TimelineRow } from "./selectors";

export type EventKind = "cross" | "peak" | "drop";

export interface DerivedEvent {
  kind: EventKind;
  label: string;
  layer: number | null;
  time: number;
  group: number;
  strength: number;
  threshold: number;
  text: string;
}

export interface EventOptions {
  /** absolute strength threshold; default = `thresholdFraction` * model max */
  threshold?: number;
  thresholdFraction?: number;
}

const fmtStrength = (v: number) => v.toFixed(2);
const fmtTime = (t: number) => `${t.toFixed(1)}s`;
const fmtLayer = (l: number | null) => (l == null ? "layer n/a" : `layer ${l}`);

function phrase(
  kind: EventKind,
  label: string,
  layer: number | null,
  time: number,
  strength: number,
  threshold: number,
): string {
  const verb =
    kind === "cross"
      ? `crosses strength ${fmtStrength(threshold)}`
      : kind === "drop"
        ? `drops below strength ${fmtStrength(threshold)}`
        : `peaks (strength ${fmtStrength(strength)})`;
  return `${label} readout ${verb} at ${fmtLayer(layer)}, t=${fmtTime(time)}`;
}

function eventsForRow(
  row: TimelineRow,
  threshold: number,
  groupTime: (i: number) => number,
): DerivedEvent[] {
  const out: DerivedEvent[] = [];
  const n = row.cells.length;
  let above = false;

  const push = (kind: EventKind, i: number, strength: number) => {
    const layer = row.layers[i] ?? row.peakLayer;
    const time = groupTime(i);
    out.push({
      kind,
      label: row.label,
      layer,
      time,
      group: i,
      strength,
      threshold,
      text: phrase(kind, row.label, layer, time, strength, threshold),
    });
  };

  for (let i = 0; i < n; i++) {
    const val = row.cells[i] ?? 0;

    if (!above && val >= threshold) {
      above = true;
      push("cross", i, val);
    } else if (above && val < threshold) {
      above = false;
      push("drop", i, val);
    }

    const prev = i > 0 ? row.cells[i - 1] ?? 0 : -Infinity;
    const next = i < n - 1 ? row.cells[i + 1] ?? 0 : -Infinity;
    if (val >= threshold && val >= prev && val > next && val > 0) {
      push("peak", i, val);
    }
  }
  return out;
}

/** Derive all events across a timeline model, ordered by time. */
export function deriveEvents(
  model: TimelineModel,
  opts: EventOptions = {},
): DerivedEvent[] {
  const frac = opts.thresholdFraction ?? 0.5;
  const threshold = opts.threshold ?? model.maxStrength * frac;
  const groupTime = (i: number) => model.groups[i]?.time_start ?? 0;

  const events: DerivedEvent[] = [];
  for (const row of model.rows) {
    events.push(...eventsForRow(row, threshold, groupTime));
  }
  events.sort((a, b) => a.time - b.time || b.strength - a.strength);
  return events;
}
