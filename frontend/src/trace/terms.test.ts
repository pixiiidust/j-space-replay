import { describe, it, expect } from "vitest";
import { answerGridPulse, canonWord, contentWords, contradictedTerms, tokensHitTerms } from "./terms";
import type { AnswerToken } from "./types";

describe("contradiction tracking (adversarial signal)", () => {
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

  it("detects a negated question term in the answer", () => {
    const t = contradictedTerms(
      "Why is the dog wet?",
      "The image does not show a dog or any indication of why a dog might be wet.",
    );
    expect(t.has("dog")).toBe(true);
  });

  it("no negation, no contradiction", () => {
    const t = contradictedTerms(
      "Why does the car start moving?",
      "The car starts moving because the traffic light changes from red to green.",
    );
    expect(t.size).toBe(0);
  });

  it("negation of a non-question word does not fire", () => {
    const t = contradictedTerms("Why is the dog wet?", "There is no umbrella in the scene.");
    expect(t.size).toBe(0);
  });

  it("pulse matrix lights cells whose readouts contain a contradicted term", () => {
    const terms = new Set(["dog"]);
    const tokens: AnswerToken[] = [
      { token: " no", readouts_by_layer: { "27": { top_tokens: [" dog"], strengths: [1] } } },
      { token: " ball", readouts_by_layer: { "27": { top_tokens: [" ball"], strengths: [1] } } },
    ];
    const pulse = answerGridPulse(tokens, [27], terms);
    expect(pulse[0][0]).toBe(true);
    expect(pulse[1][0]).toBe(false);
    expect(tokensHitTerms([" dogs"], terms)).toBe(true); // canonical match
  });
});
