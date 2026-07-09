import { useMemo, useState } from "react";
import type { Trace } from "../trace/types";
import { answerLayerGrid, answerLayerWords, answerTokensAt, type TokenStrength } from "../trace/jspace";
import { LENS_LABELS, STRENGTH_AXIS_LABEL } from "../constants";
import { answerGridPulse, deniedTerms } from "../trace/terms";
import { GridCanvas } from "./GridCanvas";

export function LensChip({ lens, caveats }: { lens: string; caveats?: string[] }) {
  const pretty = LENS_LABELS[lens] ?? lens;
  return (
    <span
      className={"lens-chip" + (lens === "j-lens-v1" ? " jlens" : "")}
      title={(caveats ?? []).join("\n") || `readouts decoded with ${pretty}`}
    >
      {pretty}
    </span>
  );
}

function PulseLegend({ terms }: { terms: Set<string> }) {
  if (terms.size === 0) return null;
  return (
    <span style={{ color: "#c0392b" }}>
      {" "}· pulsing cells read “{[...terms].join('”, “')}” — content the answer denies
    </span>
  );
}

function DrillTable({ title, tokens, live }: { title: string; tokens: TokenStrength[]; live: boolean }) {
  return (
    <div>
      <div className="panel-h" style={{ border: "none", padding: "2px 0" }}>
        <span>{title}</span>
        {live && <span style={{ color: "#113f8c" }}>● live</span>}
      </div>
      <table className="drill">
        <thead>
          <tr>
            <th>#</th>
            <th>raw token</th>
            <th className="num">{STRENGTH_AXIS_LABEL}</th>
          </tr>
        </thead>
        <tbody>
          {tokens.length === 0 && (
            <tr>
              <td colSpan={3} className="muted">no readout at this cell</td>
            </tr>
          )}
          {tokens.map((t, i) => (
            <tr key={i}>
              <td className="muted">{i + 1}</td>
              <td>{t.token}</td>
              <td className="num">{t.strength.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * The hero surface: answer-token × layer, replayed on the video clock. Token
 * rows reveal proportionally as the clip plays — the answer visibly forming.
 *
 * The drill table (pinned right) is live during playback: it streams the raw
 * top-10 of the token currently being formed, at the selected layer (click a
 * cell to choose the layer). Paused, clicking any cell inspects exactly that
 * cell — the honest "drop to raw readouts" path behind every derived view.
 */
export function AnswerWorkspace({
  trace,
  answerRow,
  playing,
}: {
  trace: Trace;
  answerRow: number;
  playing: boolean;
}) {
  const al = useMemo(() => answerLayerGrid(trace), [trace]);
  const words = useMemo(() => answerLayerWords(trace), [trace]);
  const lastLayerCol = al.layers.length - 1;
  const [sel, setSel] = useState<{ r: number; c: number } | null>(
    trace.answer_tokens.length ? { r: 0, c: lastLayerCol } : null,
  );
  const rowLabels = trace.answer_tokens.map((t, i) => `${i}:${t.token.replace(/▁/g, "·")}`);
  const terms = useMemo(() => deniedTerms(trace.answer), [trace]);
  const pulse = useMemo(
    () => answerGridPulse(trace.answer_tokens, al.layers, terms),
    [trace, al.layers, terms],
  );

  // during playback the drill follows the forming token at the chosen layer;
  // paused, it shows whatever cell was clicked
  const target = playing
    ? { r: answerRow, c: sel?.c ?? lastLayerCol }
    : sel ?? { r: answerRow, c: lastLayerCol };

  const drill = useMemo(() => {
    const at = trace.answer_tokens[target.r];
    const layer = al.layers[target.c];
    if (!at) return { title: "—", tokens: [] as TokenStrength[] };
    return {
      title: `answer token "${at.token}" · layer ${layer}`,
      tokens: answerTokensAt(at, layer),
    };
  }, [target.r, target.c, trace, al.layers]);

  return (
    <div className="panel">
      <div className="panel-h">
        <span>
          Workspace Slice · answer-token × layer{" "}
          <LensChip lens={trace.meta.lens} caveats={trace.meta.lens_caveats} />
        </span>
        <span className="muted">
          {playing ? "playing — click a column to pick the layer" : "click a cell → raw top-10"}
        </span>
      </div>
      <div className="panel-b hero-split">
        <div className="hero-grid">
          <div className="axis-note" style={{ marginBottom: 3 }}>
            each cell prints its top-1 readout; color = {STRENGTH_AXIS_LABEL} (raw
            unembedding score) — clicking a cell emphasizes every cell reading the
            same word; generation replayed on the clip clock
            <PulseLegend terms={terms} />
          </div>
          <GridCanvas
            rowLabels={rowLabels}
            colLabels={al.layers.map(String)}
            values={al.values}
            cellW={84}
            cellH={20}
            cellText={words}
            selected={target}
            highlightRow={answerRow}
            revealUpToRow={answerRow}
            pulse={pulse}
            onPick={(r, c) => setSel({ r, c })}
          />
        </div>
        <div className="hero-drill">
          <DrillTable title={drill.title} tokens={drill.tokens} live={playing} />
        </div>
      </div>
    </div>
  );
}
