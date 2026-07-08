import { describe, it, expect } from "vitest";
import { answerGridMarks, canonWord, contentWords, markFor } from "./terms";
import type { AnswerToken } from "./types";

describe("term tracking (adversarial signal)", () => {
  it("extracts canonical content words, dropping stopwords", () => {
    const w = contentWords("Why is the dog wet in this video?");
    expect(w.has("dog")).toBe(true);
    expect(w.has("wet")).toBe(true);
    expect(w.has("the")).toBe(false);
    expect(w.has("why")).toBe(false);
  });

  it("canonWord strips trivial inflections", () => {
    expect(canonWord("dogs")).toBe("dog");
    expect(canonWord("falling")).toBe("fall");
  });

  it("marks cells by which side's terms their readouts contain", () => {
    const q = contentWords("Why is the dog wet?");
    const a = contentWords("This is a cat, not a dog. It is dry.");
    expect(markFor([" cat"], q, a)).toBe("a");
    expect(markFor([" wet"], q, a)).toBe("q");
    expect(markFor([" dog"], q, a)).toBe("qa"); // in both question and answer
    expect(markFor([" zebra"], q, a)).toBe(null);
  });

  it("false-presupposition case: question terms scarce, answer correction everywhere", () => {
    const q = contentWords("Why is the dog wet?");
    const a = contentWords("The video shows a cat.");
    const tokens: AnswerToken[] = [
      { token: " cat", readouts_by_layer: { "27": { top_tokens: [" cat"], strengths: [1] } } },
      { token: " sits", readouts_by_layer: { "27": { top_tokens: [" chair"], strengths: [1] } } },
    ];
    const marks = answerGridMarks(tokens, [27], q, a);
    expect(marks[0][0]).toBe("a"); // the correction lights as an answer term
    expect(marks[1][0]).toBe(null); // no dog/wet marks anywhere
  });
});
