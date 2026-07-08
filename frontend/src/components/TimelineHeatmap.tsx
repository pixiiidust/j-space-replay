import { useEffect, useMemo, useRef, useState } from "react";
import type { TimelineModel } from "../trace/selectors";
import { STRENGTH_AXIS_LABEL } from "../constants";
import { heatColor, heatTextColor, setupCanvas } from "./canvas";

const LABEL_W = 128;
const HEADER_H = 22;
const CELL_H = 20;
const MIN_CELL_W = 52;

interface Hover {
  row: number;
  col: number;
  x: number;
  y: number;
}

interface Props {
  model: TimelineModel;
  currentGroup: number;
  selectedRow: number | null;
  onPick(rowIdx: number, groupIdx: number): void;
}

/**
 * Main timeline surface: concepts (rows) x frame groups (columns), cell = the
 * row's readout strength that group. Hover inspects; click selects the row and
 * seeks the clock to that group. Rendered on <canvas> — cell counts make DOM
 * grids sluggish.
 */
export function TimelineHeatmap({ model, currentGroup, selectedRow, onPick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<Hover | null>(null);

  const nGroups = model.groups.length;
  const nRows = model.rows.length;
  const cellW = Math.max(MIN_CELL_W, Math.floor(560 / Math.max(1, nGroups)));
  const width = LABEL_W + nGroups * cellW;
  const height = HEADER_H + nRows * CELL_H;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = setupCanvas(canvas, width, height);
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.font = "11px 'IBM Plex Mono', monospace";
    ctx.textBaseline = "middle";

    // header: group index + time
    ctx.fillStyle = "#8b8b84";
    for (let c = 0; c < nGroups; c++) {
      const x = LABEL_W + c * cellW;
      const g = model.groups[c];
      ctx.textAlign = "center";
      ctx.fillText(`g${g.group}`, x + cellW / 2, HEADER_H / 2 - 5);
      ctx.fillText(`${g.time_start}-${g.time_end}s`, x + cellW / 2, HEADER_H / 2 + 6);
    }

    // current group column highlight
    if (currentGroup >= 0 && currentGroup < nGroups) {
      ctx.strokeStyle = "#113f8c";
      ctx.lineWidth = 2;
      ctx.strokeRect(LABEL_W + currentGroup * cellW + 1, HEADER_H + 1, cellW - 2, nRows * CELL_H - 2);
    }

    // cells + labels
    for (let r = 0; r < nRows; r++) {
      const row = model.rows[r];
      const y = HEADER_H + r * CELL_H;

      // row label
      ctx.textAlign = "left";
      ctx.fillStyle = selectedRow === r ? "#113f8c" : "#1d1d1b";
      const label = row.label.length > 15 ? row.label.slice(0, 14) + "…" : row.label;
      ctx.fillText(label, 6, y + CELL_H / 2);

      for (let c = 0; c < nGroups; c++) {
        const v = row.cells[c];
        const x = LABEL_W + c * cellW;
        const t = v == null ? 0 : v / model.maxStrength;
        ctx.fillStyle = v == null ? "#f7f7f5" : heatColor(t);
        ctx.fillRect(x + 1, y + 1, cellW - 2, CELL_H - 2);
        if (v != null && v > 0) {
          ctx.fillStyle = heatTextColor(t);
          ctx.textAlign = "center";
          ctx.fillText(v.toFixed(2), x + cellW / 2, y + CELL_H / 2);
        }
      }
    }

    // selected row outline
    if (selectedRow != null && selectedRow >= 0 && selectedRow < nRows) {
      ctx.strokeStyle = "#113f8c";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(1, HEADER_H + selectedRow * CELL_H + 0.5, width - 2, CELL_H - 1);
    }
  }, [model, currentGroup, selectedRow, width, height, cellW, nGroups, nRows]);

  const cellFromEvent = (e: React.MouseEvent): Hover | null => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (x < LABEL_W || y < HEADER_H) return null;
    const col = Math.floor((x - LABEL_W) / cellW);
    const row = Math.floor((y - HEADER_H) / CELL_H);
    if (col < 0 || col >= nGroups || row < 0 || row >= nRows) return null;
    return { row, col, x, y };
  };

  const hovered = useMemo(() => {
    if (!hover) return null;
    const row = model.rows[hover.row];
    const g = model.groups[hover.col];
    return {
      label: row.label,
      strength: row.cells[hover.col],
      layer: row.layers[hover.col],
      group: g.group,
      time: `${g.time_start}-${g.time_end}s`,
      src: row.sourceTokens,
    };
  }, [hover, model]);

  return (
    <div className="panel">
      <div className="panel-h">
        <span>J-Space Timeline · {model.mode === "concepts" ? "concept" : "wordlike token"} × frame group</span>
        <span className="muted">{STRENGTH_AXIS_LABEL}</span>
      </div>
      <div className="panel-b">
        <div className="canvas-scroll" style={{ position: "relative" }}>
          <canvas
            ref={canvasRef}
            style={{ cursor: "pointer" }}
            onMouseMove={(e) => setHover(cellFromEvent(e))}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => {
              const h = cellFromEvent(e);
              if (h) onPick(h.row, h.col);
            }}
          />
          {hover && hovered && (
            <div
              className="panel"
              style={{
                position: "absolute",
                left: Math.min(hover.x + 12, width - 190),
                top: hover.y + 12,
                width: 190,
                padding: 6,
                fontSize: 10.5,
                pointerEvents: "none",
                zIndex: 5,
              }}
            >
              <div>
                <b>{hovered.label}</b>
              </div>
              <div className="muted">
                frame group g{hovered.group} · {hovered.time}
              </div>
              <div className="muted">
                layer {hovered.layer ?? "n/a"} ·{" "}
                {STRENGTH_AXIS_LABEL} {hovered.strength == null ? "—" : hovered.strength.toFixed(3)}
              </div>
              <div className="muted">src: {hovered.src.join(" ") || "—"}</div>
            </div>
          )}
        </div>
        <div className="axis-note">
          rows: {model.mode === "concepts" ? "derived concepts" : "wordlike tokens (no concept labels in this trace)"} ·
          columns: frame groups · cell = {STRENGTH_AXIS_LABEL} (click to inspect)
        </div>
      </div>
    </div>
  );
}
