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
import { GridCanvas } from "./GridCanvas";

type Sel =
  | { view: "group"; r: number; c: number }
  | { view: "answer"; r: number; c: number }
  | null;

interface Props {
  trace: Trace;
  currentGroup: number;
}

/**
 * J-space slice viewer: frame-group x layer and answer-token x layer grids.
 * Any cell drills down to its top-10 raw lens tokens. This is the honest
 * "drop to raw readouts" path SPEC requires behind every derived view.
 */
export function JSpaceSlice({ trace, currentGroup }: Props) {
  const gl = useMemo(() => groupLayerGrid(trace), [trace]);
  const al = useMemo(() => answerLayerGrid(trace), [trace]);
  const [sel, setSel] = useState<Sel>({ view: "group", r: 0, c: gl.layers.length - 1 });

  const groupRowLabels = trace.frame_groups.map((g) => `g${g.group} ${g.time_start}s`);
  const answerRowLabels = trace.answer_tokens.map((t, i) => `${i}:${t.token.replace(/▁/g, "·")}`);

  const drill: { title: string; tokens: TokenStrength[] } = useMemo(() => {
    if (!sel) return { title: "—", tokens: [] };
    if (sel.view === "group") {
      const g = trace.frame_groups[sel.r];
      const layer = gl.layers[sel.c];
      return {
        title: `frame group g${g?.group} · layer ${layer}`,
        tokens: rawTokensAt(g, layer),
      };
    }
    const at = trace.answer_tokens[sel.r];
    const layer = al.layers[sel.c];
    return {
      title: `answer token "${at?.token}" · layer ${layer}`,
      tokens: answerTokensAt(at, layer),
    };
  }, [sel, trace, gl.layers, al.layers]);

  return (
    <div className="panel">
      <div className="panel-h">
        <span>Workspace Slice · layer × position</span>
        <span className="muted">click a cell → raw top-10</span>
      </div>
      <div className="panel-b" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div>
          <div className="axis-note" style={{ marginBottom: 3 }}>
            frame-group × layer ({STRENGTH_AXIS_LABEL}, patch-share)
          </div>
          <GridCanvas
            rowLabels={groupRowLabels}
            colLabels={gl.layers.map(String)}
            values={gl.values}
            selected={sel?.view === "group" ? { r: sel.r, c: sel.c } : null}
            highlightRow={currentGroup}
            onPick={(r, c) => setSel({ view: "group", r, c })}
          />
        </div>

        {trace.answer_tokens.length > 0 && (
          <div>
            <div className="axis-note" style={{ marginBottom: 3 }}>
              answer-token × layer ({STRENGTH_AXIS_LABEL}, raw logit)
            </div>
            <GridCanvas
              rowLabels={answerRowLabels}
              colLabels={al.layers.map(String)}
              values={al.values}
              cellH={13}
              selected={sel?.view === "answer" ? { r: sel.r, c: sel.c } : null}
              onPick={(r, c) => setSel({ view: "answer", r, c })}
            />
          </div>
        )}

        <div>
          <div className="panel-h" style={{ border: "none", padding: "2px 0" }}>
            {drill.title}
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
              {drill.tokens.length === 0 && (
                <tr>
                  <td colSpan={3} className="muted">
                    no readout at this cell
                  </td>
                </tr>
              )}
              {drill.tokens.map((t, i) => (
                <tr key={i}>
                  <td className="muted">{i + 1}</td>
                  <td>{t.token}</td>
                  <td className="num">{t.strength.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
