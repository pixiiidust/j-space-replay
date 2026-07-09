import { describe, it, expect } from "vitest";
import { unspokenReadouts } from "./unspoken";
import { makeTrace } from "./testTrace";
import type { AnswerToken } from "./types";

function readout(tokens: string[]) {
  return { top_tokens: tokens, strengths: tokens.map(() => 1) };
}

describe("unspoken readouts (suppression signal)", () => {
  const answerTokens: AnswerToken[] = [0, 1, 2].map((i) => ({
    token: ` t${i}`,
    readouts_by_layer: {
      "20": readout([" blackmail", " the", " all"]),
      "24": readout([" blackmail", " gravity"]),
    },
  }));

  const trace = makeTrace([], {
    question: "Why does the ball fall?",
    answer: "The ball falls because of gravity.",
    answer_tokens: answerTokens,
  });

  it("ranks words read in many cells that Q&A never contain", () => {
    const u = unspokenReadouts(trace, { minCells: 3 });
    expect(u[0].word).toBe("blackmail");
    expect(u[0].cells).toBe(6); // 3 tokens x 2 layers
  });

  it("excludes answer words, question words, and filler", () => {
    const u = unspokenReadouts(trace, { minCells: 1 });
    const words = u.map((x) => x.word);
    expect(words).not.toContain("gravity"); // said in the answer
    expect(words).not.toContain("ball"); // in the question
    expect(words).not.toContain("the"); // stopword
    expect(words).not.toContain("all"); // filler
  });

  it("suffix variants and subword pieces of spoken words are excluded", () => {
    const t = makeTrace([], {
      question: "Why does the ball fall?",
      answer: "The ball falls because of a fundamental force called gravity.",
      answer_tokens: [0, 1, 2].map((i) => ({
        token: ` t${i}`,
        readouts_by_layer: { "24": readout([" forces", " grav", " blackmail"]) },
      })) as AnswerToken[],
    });
    const words = unspokenReadouts(t, { minCells: 1 }).map((x) => x.word);
    expect(words).not.toContain("forces"); // "force" is in the answer
    expect(words).not.toContain("grav"); // subword of "gravity"
    expect(words).toContain("blackmail");
  });

  it("minCells drops rare readouts", () => {
    const u = unspokenReadouts(trace, { minCells: 7 });
    expect(u.length).toBe(0);
  });
});
