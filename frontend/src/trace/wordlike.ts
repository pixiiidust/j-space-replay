/**
 * Wordlike-token filter — used for the fallback timeline/concept views when a
 * trace has no derived concept labels. Per issue #6: "filter tokens: trimmed,
 * >=2 chars, ascii alphabetic".
 *
 * Lens tokens are sentencepiece pieces that often carry a leading word-boundary
 * marker (U+2581 "▁") or a leading space. We strip those boundary markers and
 * surrounding whitespace before testing, so "▁floor" -> "floor" is wordlike,
 * while "oor", punctuation, CJK, and special tokens like "<|object_ref_start|>"
 * are rejected.
 */

/** U+2581 LOWER ONE EIGHTH BLOCK — sentencepiece word-boundary marker. */
const BOUNDARY = /^[▁\s]+/;

export function normalizeToken(token: string): string {
  return token.replace(BOUNDARY, "").trim();
}

export function isWordlike(token: string): boolean {
  const t = normalizeToken(token);
  return t.length >= 2 && /^[a-zA-Z]+$/.test(t);
}

/** Lowercased normalized form, used as the aggregation key for a token. */
export function wordKey(token: string): string {
  return normalizeToken(token).toLowerCase();
}
