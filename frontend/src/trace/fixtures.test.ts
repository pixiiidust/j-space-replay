/**
 * Integration check: the real committed fixture traces parse and drive every
 * selector without throwing, and exercise the fallback path (fixtures have no
 * concepts yet).
 */
import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Trace } from "./types";
import { buildTimeline, traceDuration } from "./selectors";
import { deriveEvents } from "./events";
import { answerLayerGrid, groupLayerGrid, patchHeatmapForWord, bestLayerForWord } from "./jspace";

const here = dirname(fileURLToPath(import.meta.url));
const tracesDir = resolve(here, "..", "..", "..", "fixtures", "traces");

const names = existsSync(tracesDir)
  ? readdirSync(tracesDir).filter((f) => f.endsWith(".trace.json"))
  : [];

describe("real fixture traces", () => {
  it("finds committed fixtures", () => {
    expect(names.length).toBeGreaterThan(0);
  });

  for (const name of names) {
    describe(name, () => {
      const trace = JSON.parse(
        readFileSync(resolve(tracesDir, name), "utf8"),
      ) as Trace;

      it("has schema v1 and non-empty frame groups", () => {
        expect(trace.schema).toBe(1);
        expect(trace.frame_groups.length).toBeGreaterThan(0);
        expect(traceDuration(trace)).toBeGreaterThan(0);
      });

      it("builds a fallback timeline with wordlike rows", () => {
        const m = buildTimeline(trace);
        expect(m.mode).toBe("fallback"); // fixtures have empty concepts
        expect(m.rows.length).toBeGreaterThan(0);
        for (const row of m.rows) {
          expect(row.label).toMatch(/^[a-zA-Z]{2,}$/);
        }
      });

      it("derives measurement-phrased events only", () => {
        const events = deriveEvents(buildTimeline(trace));
        for (const e of events) {
          expect(e.text).not.toMatch(/confidence|probability|realiz/i);
        }
      });

      it("builds j-space grids and a patch heatmap", () => {
        const gl = groupLayerGrid(trace);
        expect(gl.layers.length).toBe(trace.meta.n_layers);
        const al = answerLayerGrid(trace);
        expect(al.values.length).toBe(trace.answer_tokens.length);

        const row = buildTimeline(trace).rows[0];
        const g0 = trace.frame_groups[0];
        const layer = bestLayerForWord(g0, row.label) ?? gl.layers[0];
        const hm = patchHeatmapForWord(g0, layer, trace.meta.token_strings, row.label);
        // patch_top1 present in fixtures -> heatmap should build
        if (hm) {
          expect(hm.rows * hm.cols).toBe(hm.values.length);
        }
      });
    });
  }
});
