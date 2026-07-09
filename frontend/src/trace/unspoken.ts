/**
 * Unspoken readouts — the workspace paper's suppression signal, adapted.
 *
 * The paper's headline mismatch is content the workspace READS while the
 * output never says it ("strategic deliberations ... surface in the
 * workspace even when not explicit in the model's outputs"). Marking every
 * non-surfacing readout would flag nearly every cell, so this surfaces only
 * the salient tail: content words read across MANY cells that appear in
 * neither the question nor the answer.
 *
 * Mechanical string analysis of decoded tokens — never a claim about model
 * beliefs. Ranked by how many cells read the word; echoes of the question
 * are excluded because reading the question is normal processing, not
 * suppression.
 */
import type { Trace } from "./types";
import { STOP, canonWord, contentWords } from "./terms";
import { isWordlike, normalizeToken } from "./wordlike";

export interface UnspokenWord {
  /** display form (most common surface form seen) */
  word: string;
  /** distinct (position, layer) cells whose top readouts contain it */
  cells: number;
}

/**
 * Function/filler words beyond terms.STOP. Mid-layer readouts contain these
 * in hundreds of cells ("all", "only", "which" ...) — without this filter
 * they drown every real unspoken word. Kept local so deniedTerms (#29)
 * semantics are untouched.
 */
const FILLER = new Set([
  "all", "only", "just", "more", "most", "some", "any", "each", "both",
  "very", "also", "then", "here", "now", "well", "still", "even", "much",
  "many", "other", "another", "same", "such", "own", "one", "two", "three",
  "first", "last", "next", "new", "which", "while", "because", "about",
  "after", "before", "again", "against", "between", "through", "during",
  "these", "those", "they", "them", "their", "she", "him", "her", "his",
  "its", "you", "your", "will", "would", "can", "could", "may", "might",
  "must", "shall", "should", "seem", "appear", "look", "like", "way",
  "thing", "part", "side", "kind", "sort", "bit", "lot",
]);

/**
 * Content words the trace reads but the (question + answer) never contain.
 * Scans the answer-token grid (top `k` per cell) and the visual patch grid
 * (top 3 per cell). Words read in fewer than `minCells` cells are dropped.
 */
export function unspokenReadouts(
  trace: Trace,
  { k = 5, minCells = 3, limit = 8 }: { k?: number; minCells?: number; limit?: number } = {},
): UnspokenWord[] {
  const spoken = new Set<string>([
    ...contentWords(trace.question),
    ...contentWords(trace.answer),
  ]);
  const counts = new Map<string, { cells: number; surface: string }>();

  const bump = (tokens: string[]) => {
    const seen = new Set<string>();
    for (const tok of tokens) {
      if (!isWordlike(tok)) continue;
      const surface = normalizeToken(tok);
      if (surface.length < 3) continue;
      const lower = surface.toLowerCase();
      const c = canonWord(lower);
      if (STOP.has(lower) || FILLER.has(lower) || FILLER.has(c) || spoken.has(c) || seen.has(c)) continue;
      seen.add(c);
      const cur = counts.get(c);
      if (cur) cur.cells += 1;
      else counts.set(c, { cells: 1, surface: surface.toLowerCase() });
    }
  };

  for (const at of trace.answer_tokens) {
    for (const r of Object.values(at.readouts_by_layer)) bump(r.top_tokens.slice(0, k));
  }
  for (const g of trace.frame_groups) {
    for (const r of g.raw_readouts) bump(r.top_tokens.slice(0, 3));
  }

  return [...counts.values()]
    .filter((e) => e.cells >= minCells)
    .sort((a, b) => b.cells - a.cells)
    .slice(0, limit)
    .map((e) => ({ word: e.surface, cells: e.cells }));
}
