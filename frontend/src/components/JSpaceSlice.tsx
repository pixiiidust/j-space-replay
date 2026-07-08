import { useMemo, useState } from "react";
import type { Trace } from "../trace/types";
import {
  answerLayerGrid,
  answerTokensAt,
  groupLayerGrid,
  rawTokensAt,
  type TokenStrength,
} from "../trace/jspace";
import { STRENGTH_AXIS_LABEL } from "../constants";
import { answerGridMarks, groupGridMarks, questionAnswerWords } from "../trace/terms";
import { GridCanvas } from "./GridCanvas";

function MarkLegend() {
  return (
    <span>
      {" "}· <span style={{ color: "#9b2f5f" }}>◥ question term</span>{" "}
      <span style={{ color: "#caa24a" }}>◢ answer term</span> in the cell's raw readouts —
      a question whose premise the clip contradicts shows few ◥ and many ◢
    </span>
  );
}

function DrillTable({ title, tokens }: { title: string; tokens: TokenStrength[] }) {
  return (
    <div>
      <div className="panel-h" style={{ border: "none", padding: "2px 0" }}>{title}</div>
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
 * Any cell drills down to its raw top-10 lens tokens (the honest "drop to raw
 * readouts" path SPEC requires behind every derived view).
 */
export function AnswerWorkspace({
  trace,
  answerRow,
}: {
  trace: Trace;
  answerRow: number;
}) {
  const al = useMemo(() => answerLayerGrid(trace), [trace]);
  const [sel, setSel] = useState<{ r: number; c: number } | null>(
    trace.answer_tokens.length ? { r: 0, c: al.layers.length - 1 } : null,
  );
  const rowLabels = trace.answer_tokens.map((t, i) => `${i}:${t.token.replace(/▁/g, "·")}`);
  const marks = useMemo(() => {
    const { qWords, aWords } = questionAnswerWords(trace);
    return answerGridMarks(trace.answer_tokens, al.layers, qWords, aWords);
  }, [trace, al.layers]);

  const drill = useMemo(() => {
    if (!sel) return { title: "—", tokens: [] as TokenStrength[] };
    const at = trace.answer_tokens[sel.r];
    const layer = al.layers[sel.c];
    return {
      title: `answer token "${at?.token}" · layer ${layer}`,
      tokens: answerTokensAt(at, layer),
    };
  }, [sel, trace, al.layers]);

  return (
    <div className="panel">
      <div className="panel-h">
        <span>Workspace Slice · answer-token × layer</span>
        <span className="muted">click a cell → raw top-10</span>
      </div>
      <div className="panel-b hero-split">
        <div className="hero-grid">
          <div className="axis-note" style={{ marginBottom: 3 }}>
            {STRENGTH_AXIS_LABEL}, raw logit — generation replayed on the clip clock;
            tokens were generated after the model saw the whole clip
            <MarkLegend />
          </div>
          <GridCanvas
            rowLabels={rowLabels}
            colLabels={al.layers.map(String)}
            values={al.values}
            cellH={13}
            selected={sel}
            highlightRow={answerRow}
            revealUpToRow={answerRow}
            marks={marks}
            onPick={(r, c) => setSel({ r, c })}
          />
        </div>
        <div className="hero-drill">
          <DrillTable title={drill.title} tokens={drill.tokens} />
        </div>
      </div>
    </div>
  );
}

/**
 * Frame-group × layer, transposed (layers as rows, deep layers on top; frame
 * groups as columns) so time runs left-to-right beside the video and the
 * panel fits a narrow column. Columns light up as the playhead passes them;
 * clicking a column seeks the clip to that group.
 */
export function GroupLayerPanel({
  trace,
  currentGroup,
  onSeekGroup,
}: {
  trace: Trace;
  currentGroup: number;
  onSeekGroup?(idx: number): void;
}) {
  const gl = useMemo(() => groupLayerGrid(trace), [trace]);
  const nLayers = gl.layers.length;
  // row r shows layer nLayers-1-r (deep layers, where the signal lives, on top)
  const values = useMemo(
    () =>
      Array.from({ length: nLayers }, (_, r) =>
        trace.frame_groups.map((_, g) => gl.values[g][nLayers - 1 - r]),
      ),
    [gl, nLayers, trace.frame_groups],
  );
  const [sel, setSel] = useState<{ r: number; c: number } | null>({ r: 0, c: 0 });
  const marks = useMemo(() => {
    const { qWords, aWords } = questionAnswerWords(trace);
    return groupGridMarks(trace.frame_groups, nLayers, qWords, aWords);
  }, [trace, nLayers]);

  const drill = useMemo(() => {
    if (!sel) return { title: "—", tokens: [] as TokenStrength[] };
    const layer = nLayers - 1 - sel.r;
    const g = trace.frame_groups[sel.c];
    return {
      title: `g${g?.group} (${g?.time_start}–${g?.time_end}s) · layer ${layer}`,
      tokens: rawTokensAt(g, layer),
    };
  }, [sel, trace, nLayers]);

  return (
    <div className="panel">
      <div className="panel-h">
        <span>Frame Group × Layer</span>
        <span className="muted">click → raw top-10</span>
      </div>
      <div className="panel-b" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <div>
          <div className="axis-note" style={{ marginBottom: 3 }}>
            {STRENGTH_AXIS_LABEL}, patch-share — rows are layers (deep on top), columns
            are frame groups on the clip timeline
            <MarkLegend />
          </div>
          <GridCanvas
            rowLabels={gl.layers.map((l) => String(l)).reverse()}
            colLabels={trace.frame_groups.map((g) => `g${g.group}`)}
            values={values}
            labelW={34}
            cellW={22}
            cellH={13}
            selected={sel}
            highlightCol={currentGroup}
            revealUpToCol={currentGroup}
            marks={marks}
            onPick={(r, c) => {
              setSel({ r, c });
              onSeekGroup?.(c);
            }}
          />
        </div>
        <DrillTable title={drill.title} tokens={drill.tokens} />
      </div>
    </div>
  );
}
