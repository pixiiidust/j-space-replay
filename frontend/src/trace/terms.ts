/**
 * Contradiction tracking — the "adversarial signal" surface.
 *
 * When the answer NEGATES one of the question's content words ("why is the
 * dog wet?" → "the clip does not show a dog"), every cell whose raw readouts
 * contain that word pulses red: the workspace visibly handling the rejected
 * premise. Detection is purely mechanical string analysis (negation cue near
 * a question word in the answer; readout string match) — never a claim about
 * model beliefs.
 */
import type { AnswerToken } from "./types";

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

const NEGATIONS = new Set([
  "no", "not", "never", "without", "isn't", "aren't", "wasn't", "doesn't",
  "don't", "didn't", "cannot", "can't", "nor", "none",
]);

/**
 * Question content words that the answer NEGATES: the word appears in the
 * answer within a few words after a negation cue ("does not show a dog",
 * "there is no dog"). Purely mechanical string analysis of the answer text.
 */
export function contradictedTerms(question: string, answer: string, window = 6): Set<string> {
  const qWords = contentWords(question);
  const out = new Set<string>();
  const words = answer.toLowerCase().match(/[a-z][a-z'-]*/g) ?? [];
  for (let i = 0; i < words.length; i++) {
    if (!NEGATIONS.has(words[i])) continue;
    for (let j = i + 1; j <= Math.min(i + window, words.length - 1); j++) {
      const c = canonWord(words[j]);
      if (qWords.has(c)) out.add(c);
    }
  }
  return out;
}

export function tokensHitTerms(tokens: string[], terms: Set<string>): boolean {
  if (terms.size === 0) return false;
  return tokens.some((t) => {
    const c = canonWord(t.trim());
    return c.length >= 3 && terms.has(c);
  });
}

/** Pulse matrix for the answer-token × layer grid: rows = tokens, cols = layers. */
export function answerGridPulse(
  answerTokens: AnswerToken[],
  layers: number[],
  terms: Set<string>,
): boolean[][] {
  return answerTokens.map((at) =>
    layers.map((layer) => {
      const r = at.readouts_by_layer[String(layer)];
      return r ? tokensHitTerms(r.top_tokens, terms) : false;
    }),
  );
}

