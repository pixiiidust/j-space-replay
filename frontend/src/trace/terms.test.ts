import { describe, it, expect } from "vitest";
import { answerGridPulse, canonWord, contentWords, deniedTerms, tokensHitTerms } from "./terms";
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

  it("detects a term the answer denies, regardless of the question", () => {
    const t = deniedTerms("The panda did not fall from the structure.");
    expect(t.has("fall")).toBe(true);
    expect(t.has("panda")).toBe(false); // the subject is not the denial
  });

  it("skips hedge scaffolding and reaches the denial's target", () => {
    // real case from user QA: the old 6-word window flagged
    // visible/evidence/being and missed the actual target "pushed"
    const t = deniedTerms(
      "The image shows a panda in an enclosure, and it appears to be climbing " +
      "or moving along a wooden structure. There is no visible evidence of the " +
      "panda being pushed. The panda seems to be actively engaging with its " +
      "environment, possibly exploring or playing.",
    );
    expect(t.has("push")).toBe(true); // canonical form of "pushed"
    expect(t.has("visible")).toBe(false);
    expect(t.has("evidence")).toBe(false);
    expect(t.has("being")).toBe(false);
    expect(t.has("panda")).toBe(false); // affirmed throughout the answer
  });

  it("no negation, no adversarial terms", () => {
    const t = deniedTerms("The car starts moving because the light changes to green.");
    expect(t.size).toBe(0);
  });

  it("denials also fire for words never in the question", () => {
    const t = deniedTerms("There is no umbrella in the scene.");
    expect(t.has("umbrella")).toBe(true);
  });

  it("a word affirmed more often than denied is dropped", () => {
    const t = deniedTerms(
      "The ball does not fall at first. Later the ball falls off the edge and keeps falling.",
    );
    expect(t.has("fall")).toBe(false);
  });

  it("pulse matrix lights cells whose readouts contain a denied term", () => {
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
