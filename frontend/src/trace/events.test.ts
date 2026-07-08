import { describe, it, expect } from "vitest";
import { deriveEvents } from "./events";
import { buildTimeline } from "./selectors";
import { group, makeTrace } from "./testTrace";

const concept = (label: string, strength: number, layer: number) => ({
  label,
  strength,
  layer,
  source_tokens: [label],
});

// strength series for "x": 0.1, 0.6, 0.9, 0.2  (threshold 0.5)
const t = makeTrace([
  group(0, { concepts: [concept("x", 0.1, 5)] }),
  group(1, { concepts: [concept("x", 0.6, 12)] }),
  group(2, { concepts: [concept("x", 0.9, 24)] }),
  group(3, { concepts: [concept("x", 0.2, 8)] }),
]);

describe("event derivation", () => {
  const model = buildTimeline(t);
  const events = deriveEvents(model, { threshold: 0.5 });

  it("emits cross, peak, and drop events with correct layers/times", () => {
    const kinds = events.map((e) => e.kind);
    expect(kinds).toContain("cross");
    expect(kinds).toContain("peak");
    expect(kinds).toContain("drop");

    const cross = events.find((e) => e.kind === "cross")!;
    expect(cross.group).toBe(1);
    expect(cross.time).toBe(2); // group 1 start
    expect(cross.layer).toBe(12);

    const peak = events.find((e) => e.kind === "peak")!;
    expect(peak.group).toBe(2);
    expect(peak.layer).toBe(24);

    const drop = events.find((e) => e.kind === "drop")!;
    expect(drop.group).toBe(3);
  });

  it("phrases events as measurements with layer + time, never narrative", () => {
    const cross = events.find((e) => e.kind === "cross")!;
    expect(cross.text).toBe("x readout crosses strength 0.50 at layer 12, t=2.0s");
    const peak = events.find((e) => e.kind === "peak")!;
    expect(peak.text).toBe("x readout peaks (strength 0.90) at layer 24, t=4.0s");

    for (const e of events) {
      expect(e.text).not.toMatch(/confidence|probability|realiz|understand|decide/i);
      expect(e.text).toMatch(/at layer .+, t=[\d.]+s$/);
    }
  });

  it("orders events by time", () => {
    const times = events.map((e) => e.time);
    const sorted = [...times].sort((a, b) => a - b);
    expect(times).toEqual(sorted);
  });
});
