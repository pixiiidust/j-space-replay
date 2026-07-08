import { describe, it, expect } from "vitest";
import {
  activeRowsAtGroup,
  buildTimeline,
  groupIndexAtTime,
  snapToGroupStart,
  stepGroupIndex,
  traceDuration,
} from "./selectors";
import { group, makeTrace } from "./testTrace";

const clock = makeTrace([group(0), group(1), group(2)]); // 0-2, 2-4, 4-6

describe("trace clock", () => {
  it("maps a video time to its frame group (half-open intervals)", () => {
    expect(groupIndexAtTime(clock, 0)).toBe(0);
    expect(groupIndexAtTime(clock, 1.9)).toBe(0);
    expect(groupIndexAtTime(clock, 2)).toBe(1);
    expect(groupIndexAtTime(clock, 4.5)).toBe(2);
  });

  it("clamps out-of-range times", () => {
    expect(groupIndexAtTime(clock, -3)).toBe(0);
    expect(groupIndexAtTime(clock, 999)).toBe(2);
  });

  it("snaps a time to its group start boundary", () => {
    expect(snapToGroupStart(clock, 3.3)).toBe(2);
    expect(snapToGroupStart(clock, 5.9)).toBe(4);
  });

  it("steps one group at a time, clamped", () => {
    expect(stepGroupIndex(clock, 0, +1)).toBe(1);
    expect(stepGroupIndex(clock, 2, +1)).toBe(2); // clamp high
    expect(stepGroupIndex(clock, 0, -1)).toBe(0); // clamp low
  });

  it("duration is the last group end", () => {
    expect(traceDuration(clock)).toBe(6);
  });
});

describe("timeline: concepts mode", () => {
  const t = makeTrace([
    group(0, {
      concepts: [{ label: "wet floor", strength: 1.0, layer: 24, source_tokens: ["▁wet", "▁floor"] }],
    }),
    group(1, {
      concepts: [{ label: "wet floor", strength: 2.7, layer: 24, source_tokens: ["▁wet", "▁floor"] }],
    }),
  ]);

  it("uses derived concepts as rows when present", () => {
    const m = buildTimeline(t);
    expect(m.mode).toBe("concepts");
    expect(m.rows).toHaveLength(1);
    const row = m.rows[0];
    expect(row.label).toBe("wet floor");
    expect(row.kind).toBe("concept");
    expect(row.cells).toEqual([1.0, 2.7]);
    expect(row.peakGroup).toBe(1);
    expect(row.peakStrength).toBeCloseTo(2.7);
    expect(row.peakLayer).toBe(24);
    expect(row.peakTime).toBe(2); // group 1 start
  });

  it("active rows at a group are sorted by strength", () => {
    const m = buildTimeline(t);
    const active = activeRowsAtGroup(m, 1);
    expect(active[0]).toEqual({ label: "wet floor", strength: 2.7 });
  });
});

describe("timeline: wordlike fallback mode", () => {
  const readout = (layer: number, toks: string[], strengths: number[]) => ({
    layer,
    top_tokens: toks,
    strengths,
  });
  const t = makeTrace([
    group(0, {
      raw_readouts: [
        readout(10, ["▁ball", "换句话", "a", "▁red"], [0.9, 0.5, 0.4, 0.3]),
      ],
    }),
    group(1, {
      raw_readouts: [
        readout(12, ["▁ball", "▁floor"], [0.4, 0.8]),
        readout(20, ["▁Ball"], [0.6]), // same word, different case + layer
      ],
    }),
  ]);

  it("falls back to wordlike tokens x frame-group and filters non-words", () => {
    const m = buildTimeline(t);
    expect(m.mode).toBe("fallback");
    const labels = m.rows.map((r) => r.label);
    expect(labels).toContain("ball");
    expect(labels).toContain("floor");
    expect(labels).toContain("red");
    expect(labels).not.toContain("换句话");
    expect(labels).not.toContain("a");
  });

  it("aggregates a word across layers by max strength and records the layer", () => {
    const m = buildTimeline(t);
    const ball = m.rows.find((r) => r.label === "ball")!;
    // group0 max = 0.9 (layer 10); group1 max = max(0.4, 0.6) = 0.6 (layer 20)
    expect(ball.cells).toEqual([0.9, 0.6]);
    expect(ball.peakGroup).toBe(0);
    expect(ball.layers[1]).toBe(20);
  });
});
