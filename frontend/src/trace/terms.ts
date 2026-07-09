/**
 * Adversarial-signal tracking: readouts that go against the model's reply.
 *
 * The workspace paper's mismatch signal is workspace-vs-OUTPUT: content that
 * surfaces in J-space readouts while the output suppresses or denies it
 * ("strategic deliberations ... surface in the workspace even when not
 * explicit in the model's outputs"). The mechanically checkable slice of
 * that here: words the ANSWER denies ("the panda did not fall", "there is
 * no dog") — every cell whose readouts contain a denied word pulses red,
 * the workspace visibly reading what the reply rejects.
 *
 * A word both denied and affirmed in the answer ("does not fall at first,
 * then falls") only counts when denials outnumber affirmations. Detection
 * is purely mechanical string analysis — never a claim about model beliefs.
 * The paper's fuller signal (any readout absent from the output) is
 * deliberately NOT marked: most readouts never surface, so it would flag
 * nearly every cell.
 */
import type { AnswerToken } from "./types";

export const STOP = new Set([
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
 * Content words the ANSWER denies: a content word inside a short window
 * after a negation cue ("did not fall", "there is no dog"). A word is
 * dropped again if the answer affirms it more often than it denies it
 * (mentions outside negation windows outnumber those inside). Purely
 * mechanical string analysis of the answer text.
 */
export function deniedTerms(answer: string, window = 6): Set<string> {
  const words = answer.toLowerCase().match(/[a-z][a-z'-]*/g) ?? [];
  // mark every position covered by a negation window
  const negated = new Array<boolean>(words.length).fill(false);
  for (let i = 0; i < words.length; i++) {
    if (!NEGATIONS.has(words[i])) continue;
    for (let j = i + 1; j <= Math.min(i + window, words.length - 1); j++) {
      negated[j] = true;
    }
  }
  const denials = new Map<string, number>();
  const affirmations = new Map<string, number>();
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (w.length < 3 || STOP.has(w) || NEGATIONS.has(w)) continue;
    const c = canonWord(w);
    const bucket = negated[i] ? denials : affirmations;
    bucket.set(c, (bucket.get(c) ?? 0) + 1);
  }
  const out = new Set<string>();
  for (const [c, n] of denials) {
    if (n > (affirmations.get(c) ?? 0)) out.add(c);
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

