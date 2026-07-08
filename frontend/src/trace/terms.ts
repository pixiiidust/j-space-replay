/**
 * Question/answer term tracking — the "adversarial signal" surface.
 *
 * Cells whose raw readouts contain the QUESTION's content words get one mark,
 * cells reading the ANSWER's content words another. When a question carries a
 * false presupposition ("why is the dog wet" on a cat video), the mismatch is
 * visible at a glance: question-term marks are scarce or absent while the
 * answer's correction ("cat") marks are everywhere. This only *marks* string
 * matches against lens readouts — it is not a claim about model beliefs.
 */
import type { AnswerToken, FrameGroup, Trace } from "./types";

const STOP = new Set([
  "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is", "are",
  "was", "were", "be", "been", "it", "its", "this", "that", "there", "with",
  "as", "by", "for", "from", "into", "over", "under", "then", "than", "but",
  "not", "no", "yes", "so", "if", "what", "which", "who", "when", "where",
  "how", "why", "does", "do", "did", "can", "could", "will", "would", "has",
  "have", "had", "video", "image", "happens", "describe", "shows",
]);

/** Canonical word form: lowercase, trivial inflection suffixes stripped. */
export function canonWord(w: string): string {
  let s = w.toLowerCase();
  for (const suf of ["ing", "ed", "es", "s"]) {
    if (s.endsWith(suf) && s.length - suf.length >= 3) {
      s = s.slice(0, -suf.length);
      break;
    }
  }
  return s;
}

export function contentWords(text: string): Set<string> {
  const out = new Set<string>();
  for (const m of text.matchAll(/[a-zA-Z][a-zA-Z-]{2,}/g)) {
    const w = m[0].toLowerCase();
    if (!STOP.has(w)) out.add(canonWord(w));
  }
  return out;
}

export type CellMark = "q" | "a" | "qa" | null;

export function markFor(tokens: string[], qWords: Set<string>, aWords: Set<string>): CellMark {
  let q = false;
  let a = false;
  for (const t of tokens) {
    const c = canonWord(t.trim());
    if (c.length < 3) continue;
    if (qWords.has(c)) q = true;
    if (aWords.has(c)) a = true;
    if (q && a) break;
  }
  return q && a ? "qa" : q ? "q" : a ? "a" : null;
}

export function questionAnswerWords(trace: Trace): { qWords: Set<string>; aWords: Set<string> } {
  return { qWords: contentWords(trace.question), aWords: contentWords(trace.answer) };
}

/** Marks for the answer-token × layer grid: rows = tokens, cols = layers. */
export function answerGridMarks(
  answerTokens: AnswerToken[],
  layers: number[],
  qWords: Set<string>,
  aWords: Set<string>,
): CellMark[][] {
  return answerTokens.map((at) =>
    layers.map((layer) => {
      const r = at.readouts_by_layer[String(layer)];
      return r ? markFor(r.top_tokens, qWords, aWords) : null;
    }),
  );
}

/** Marks for the transposed frame-group grid: rows = layers (deep on top), cols = groups. */
export function groupGridMarks(
  groups: FrameGroup[],
  nLayers: number,
  qWords: Set<string>,
  aWords: Set<string>,
): CellMark[][] {
  const byGroupLayer = groups.map((g) => {
    const m = new Map<number, string[]>();
    for (const r of g.raw_readouts) m.set(r.layer, r.top_tokens);
    return m;
  });
  return Array.from({ length: nLayers }, (_, r) => {
    const layer = nLayers - 1 - r;
    return groups.map((_, gi) => {
      const tokens = byGroupLayer[gi].get(layer);
      return tokens ? markFor(tokens, qWords, aWords) : null;
    });
  });
}
