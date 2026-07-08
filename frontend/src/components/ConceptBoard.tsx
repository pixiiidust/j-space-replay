import type { TimelineModel } from "../trace/selectors";
import { STRENGTH_AXIS_LABEL } from "../constants";

interface Props {
  model: TimelineModel;
  selectedRow: number | null;
  pinned: Set<string>;
  hidden: Set<string>;
  onSelect(idx: number): void;
  onJumpPeak(idx: number): void;
  onTogglePin(label: string): void;
  onToggleHide(label: string): void;
}

/**
 * Concept board: derived concepts with peak time + provenance, pin/hide, and
 * "jump to peak". Falls back to top wordlike tokens with an honest banner when
 * the trace carries no concept labels.
 */
export function ConceptBoard({
  model,
  selectedRow,
  pinned,
  hidden,
  onSelect,
  onJumpPeak,
  onTogglePin,
  onToggleHide,
}: Props) {
  // order: pinned first, then normal, hidden last
  const order = model.rows
    .map((row, idx) => ({ row, idx }))
    .sort((a, b) => {
      const pa = pinned.has(a.row.label) ? 0 : 1;
      const pb = pinned.has(b.row.label) ? 0 : 1;
      const ha = hidden.has(a.row.label) ? 1 : 0;
      const hb = hidden.has(b.row.label) ? 1 : 0;
      return ha - hb || pa - pb || b.row.peakStrength - a.row.peakStrength;
    });

  return (
    <div className="panel">
      <div className="panel-h">
        <span>Concepts</span>
        <span className="muted">{model.rows.length} rows</span>
      </div>
      <div className="panel-b" style={{ padding: 0 }}>
        {model.mode === "fallback" && (
          <div className="axis-note" style={{ padding: "6px 8px", color: "#9b2f5f" }}>
            no concept labels in this trace — showing top wordlike tokens from raw readouts
          </div>
        )}
        <div>
          {order.map(({ row, idx }) => (
            <div
              key={row.label}
              className={
                "concept-row" +
                (selectedRow === idx ? " sel" : "") +
                (hidden.has(row.label) ? " hidden" : "")
              }
              onClick={() => onSelect(idx)}
            >
              <div style={{ minWidth: 0 }}>
                <div className="concept-name">
                  {pinned.has(row.label) ? "📌 " : ""}
                  {row.label}
                </div>
                <div className="concept-meta">
                  peak {row.peakStrength.toFixed(2)} {STRENGTH_AXIS_LABEL} · layer{" "}
                  {row.peakLayer ?? "n/a"} · t={row.peakTime.toFixed(1)}s
                  {row.sourceTokens.length ? ` · src ${row.sourceTokens.join(" ")}` : ""}
                </div>
              </div>
              <div className="mini-actions" onClick={(e) => e.stopPropagation()}>
                <button title="jump to peak" onClick={() => onJumpPeak(idx)}>
                  peak
                </button>
                <button
                  title="pin"
                  className={pinned.has(row.label) ? "" : ""}
                  onClick={() => onTogglePin(row.label)}
                >
                  {pinned.has(row.label) ? "unpin" : "pin"}
                </button>
                <button title="hide" onClick={() => onToggleHide(row.label)}>
                  {hidden.has(row.label) ? "show" : "hide"}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
