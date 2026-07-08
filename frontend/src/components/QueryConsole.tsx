import { useEffect, useMemo, useState } from "react";
import type { Trace } from "../trace/types";
import { DEFAULT_QUESTION } from "../constants";
import { contradictedTerms } from "../trace/terms";

interface Props {
  trace: Trace;
  /** Index of the answer token currently being replayed (drives the underline). */
  answerRow: number;
  onSeekToken(index: number): void;
  onReAsk(videoId: string, question: string): void;
}

const isSpecial = (tok: string) => /^\s*<\|[^>]*\|>\s*$/.test(tok);

/**
 * The query console: question in, answer out — the pipeline's contract, above
 * the machinery that explains it. The answer is rendered token-by-token; the
 * token the replay is currently "forming" is highlighted in sync with the
 * workspace slice, and clicking any token seeks the clock to that point.
 */
export function QueryConsole({ trace, answerRow, onSeekToken, onReAsk }: Props) {
  const [q, setQ] = useState(trace.question);
  useEffect(() => setQ(trace.question), [trace.question]);
  const contradicted = useMemo(
    () => contradictedTerms(trace.question, trace.answer),
    [trace.question, trace.answer],
  );

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
            if (e.key === "Enter") onReAsk(trace.video_id, q.trim() || DEFAULT_QUESTION);
          }}
        />
        <button
          className="btn primary"
          onClick={() => onReAsk(trace.video_id, q.trim() || DEFAULT_QUESTION)}
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
          premise check: the answer negates “{[...contradicted].join("”, “")}” — cells
          reading it pulse red in the grids below
        </div>
      )}
    </div>
  );
}
