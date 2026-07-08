import { describe, it, expect } from "vitest";
import { isWordlike, normalizeToken, wordKey } from "./wordlike";

describe("wordlike filter", () => {
  it("strips the sentencepiece boundary marker before testing", () => {
    expect(normalizeToken("▁floor")).toBe("floor");
    expect(normalizeToken("  wet ")).toBe("wet");
  });

  it("accepts trimmed, >=2 char, ascii-alphabetic tokens", () => {
    expect(isWordlike("▁floor")).toBe(true);
    expect(isWordlike("wet")).toBe(true);
    expect(isWordlike("Ball")).toBe(true);
  });

  it("rejects short, non-alphabetic, punctuation, and special tokens", () => {
    expect(isWordlike("a")).toBe(false); // 1 char
    expect(isWordlike("▁a")).toBe(false);
    expect(isWordlike("123")).toBe(false);
    expect(isWordlike("floor2")).toBe(false);
    expect(isWordlike(" \"")).toBe(false);
    expect(isWordlike("<|object_ref_start|>")).toBe(false);
    expect(isWordlike("换句话")).toBe(false); // CJK
  });

  it("wordKey lowercases the normalized token for aggregation", () => {
    expect(wordKey("▁Floor")).toBe("floor");
    expect(wordKey("floor")).toBe("floor");
  });
});
