import { describe, it, expect } from "vitest";
import {
  answerLayerGrid,
  bestLayerForWord,
  groupLayerGrid,
  patchHeatmapForWord,
  rawTokensAt,
  topTokens,
} from "./jspace";
import { group, makeTrace } from "./testTrace";
import type { AnswerToken } from "./types";

describe("j-space slice selectors", () => {
  const t = makeTrace(
    [
      group(0, {
        patch_grid: [2, 2],
        raw_readouts: [
          { layer: 10, top_tokens: ["▁ball", "▁red"], strengths: [0.9, 0.3], patch_top1: [1, 2, 1, 3] },
          { layer: 24, top_tokens: ["▁floor"], strengths: [0.5] },
        ],
      }),
    ],
    {
      meta: {
        model: "m",
        lens: "l",
        temporal_resolution_frames: 2,
        strength_normalization: "s",
        n_layers: 28,
        token_strings: { "1": "▁ball", "2": "▁sky", "3": "wall" },
      },
      answer_tokens: [
        {
          token: "The",
          readouts_by_layer: {
            "20": { top_tokens: ["▁The", "▁A"], strengths: [8.0, 6.0] },
            "28": { top_tokens: ["▁The"], strengths: [12.0] },
          },
        } as AnswerToken,
      ],
    },
  );

  it("builds a frame-group x layer grid using top strength", () => {
    const grid = groupLayerGrid(t);
    expect(grid.layers).toEqual([10, 24]);
    expect(grid.values[0]).toEqual([0.9, 0.5]);
  });

  it("drills a cell down to raw top-k tokens", () => {
    const toks = rawTokensAt(t.frame_groups[0], 10);
    expect(toks[0]).toEqual({ token: "▁ball", strength: 0.9 });
  });

  it("builds an answer-token x layer grid across the layer union", () => {
    const grid = answerLayerGrid(t);
    expect(grid.layers).toEqual([20, 28]);
    expect(grid.values[0]).toEqual([8.0, 12.0]);
  });

  it("computes a patch heatmap for a word from patch_top1 + token_strings", () => {
    const hm = patchHeatmapForWord(t.frame_groups[0], 10, t.meta.token_strings, "ball");
    expect(hm).not.toBeNull();
    expect(hm!.rows).toBe(2);
    expect(hm!.cols).toBe(2);
    // patch_top1 = [1,2,1,3]; token id 1 -> "▁ball" -> matches "ball"
    expect(hm!.values).toEqual([1, 0, 1, 0]);
  });

  it("picks the layer where a word reads strongest", () => {
    expect(bestLayerForWord(t.frame_groups[0], "ball")).toBe(10);
  });

  it("topTokens clamps to k and tolerates undefined", () => {
    expect(topTokens(undefined)).toEqual([]);
    expect(topTokens({ top_tokens: ["a", "b", "c"], strengths: [3, 2, 1] }, 2)).toHaveLength(2);
  });
});
