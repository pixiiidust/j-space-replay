import { useEffect, useMemo, useState } from "react";
import type { Trace } from "../trace/types";
import { getLenses } from "../api/client";
import { DEFAULT_QUESTION, LENS_LABELS } from "../constants";
import { deniedTerms } from "../trace/terms";

interface Props {
  trace: Trace;
  /** Index of the answer token currently being replayed (drives the underline). */
  answerRow: number;
  onSeekToken(index: number): void;
  onReAsk(videoId: string, question: string, lens?: string): void;
}

const isSpecial = (tok: string) => /^\s*<\|[^>]*\|>\s*$/.test(tok);

/** Lens picker, shown only when this install has more than one lens fitted. */
export function LensSelect({
  value,
  onChange,
  lenses,
}: {
  value: string;
  onChange(lens: string): void;
  lenses: string[];
}) {
  if (lenses.length < 2) return null;
  return (
    <select
      className="lens-select"
      aria-label="decode lens"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      title="which readout method decodes the residuals — every trace is badged with the lens that made it"
    >
      {lenses.map((l) => (
        <option key={l} value={l}>
          {LENS_LABELS[l] ?? l}
        </option>
      ))}
    </select>
  );
}

/**
 * The query console: question in, answer out — the pipeline's contract, above
 * the machinery that explains it. The answer is rendered token-by-token; the
 * token the replay is currently "forming" is highlighted in sync with the
 * workspace slice, and clicking any token seeks the clock to that point.
 */
export function QueryConsole({ trace, answerRow, onSeekToken, onReAsk }: Props) {
  const [q, setQ] = useState(trace.question);
  const [lenses, setLenses] = useState<string[]>([]);
  const [lens, setLens] = useState(trace.meta.lens);
  useEffect(() => setQ(trace.question), [trace.question]);
  useEffect(() => {
    let alive = true;
    getLenses().then((info) => {
      if (!alive) return;
      setLenses(info.lenses);
      // re-ask defaults to the lens THIS trace was made with (if still available)
      setLens(info.lenses.includes(trace.meta.lens) ? trace.meta.lens : info.default);
    });
    return () => {
      alive = false;
    };
  }, [trace.meta.lens]);
  const contradicted = useMemo(() => deniedTerms(trace.answer), [trace.answer]);
  const ask = () => onReAsk(trace.video_id, q.trim() || DEFAULT_QUESTION, lens);

  return (
    <div className="qa">
      <div className="qrow">
        <span className="tag">Q</span>
        <input
          type="text"
          value={q}
          aria-label="new question"
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") ask();
          }}
        />
        <LensSelect value={lens} onChange={setLens} lenses={lenses} />
        <button
          className="btn primary"
          onClick={ask}
          title="re-runs the pipeline on this video"
        >
          re-ask ▶
        </button>
      </div>
      <div className="arow">
        <span className="tag">A</span>{" "}
        {trace.answer_tokens.length > 0
          ? trace.answer_tokens.map((t, i) =>
              isSpecial(t.token) ? null : (
                <span
                  key={i}
                  className={"atok" + (i === answerRow ? " on" : i > answerRow ? " future" : "")}
                  title={`answer token ${i} — click to replay from here`}
                  onClick={() => onSeekToken(i)}
                >
                  {t.token}
                </span>
              ),
            )
          : trace.answer}
      </div>
      {contradicted.size > 0 && (
        <div className="premise-check">
          adversarial check: the answer denies “{[...contradicted].join("”, “")}” — cells
          whose readouts contain it pulse red in the grid below
        </div>
      )}
    </div>
  );
}
