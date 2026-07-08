import { useEffect, useRef } from "react";
import { heatColor, setupCanvas } from "./canvas";

interface Props {
  rowLabels: string[];
  colLabels: string[];
  values: (number | null)[][]; // [row][col]
  selected: { r: number; c: number } | null;
  onPick(r: number, c: number): void;
  labelW?: number;
  cellW?: number;
  cellH?: number;
  highlightRow?: number;
  highlightCol?: number;
  /** Playback reveal: rows after this index render faint until the clock reaches them. */
  revealUpToRow?: number | null;
  /** Column-wise variant, for grids whose time axis runs left-to-right. */
  revealUpToCol?: number | null;
  /** Per-cell term marks: "q" (question term), "a" (answer term), "qa", or null. */
  marks?: (string | null)[][];
}

/**
 * Generic heat grid on <canvas>: rows x cols, each cell normalized by the
 * grid's max magnitude. Click selects a cell. Used for the J-space slices
 * (frame-group x layer, answer-token x layer).
 */
export function GridCanvas({
  rowLabels,
  colLabels,
  values,
  selected,
  onPick,
  labelW = 96,
  cellW = 24,
  cellH = 16,
  highlightRow,
  highlightCol,
  revealUpToRow,
  revealUpToCol,
  marks,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const headerH = 16;
  const nRows = rowLabels.length;
  const nCols = colLabels.length;
  const width = labelW + nCols * cellW;
  const height = headerH + nRows * cellH;

  let maxAbs = 0;
  for (const row of values) for (const v of row) if (v != null) maxAbs = Math.max(maxAbs, Math.abs(v));
  if (maxAbs === 0) maxAbs = 1;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = setupCanvas(canvas, width, height);
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.textBaseline = "middle";

    ctx.textAlign = "center";
    for (let c = 0; c < nCols; c++) {
      ctx.fillStyle = highlightCol === c ? "#113f8c" : "#8b8b84";
      ctx.fillText(colLabels[c], labelW + c * cellW + cellW / 2, headerH / 2);
    }

    for (let r = 0; r < nRows; r++) {
      const y = headerH + r * cellH;
      const unrevealed = revealUpToRow != null && r > revealUpToRow;
      ctx.textAlign = "left";
      ctx.fillStyle = highlightRow === r ? "#113f8c" : unrevealed ? "#c9c9c2" : "#1d1d1b";
      const lbl = rowLabels[r].length > 13 ? rowLabels[r].slice(0, 12) + "…" : rowLabels[r];
      ctx.fillText(lbl, 4, y + cellH / 2);
      for (let c = 0; c < nCols; c++) {
        const v = values[r][c];
        const x = labelW + c * cellW;
        const cellUnrevealed = unrevealed || (revealUpToCol != null && c > revealUpToCol);
        ctx.globalAlpha = cellUnrevealed ? 0.12 : 1;
        ctx.fillStyle = v == null ? "#f7f7f5" : heatColor(Math.abs(v) / maxAbs);
        ctx.fillRect(x + 0.5, y + 0.5, cellW - 1, cellH - 1);
        const mark = marks?.[r]?.[c];
        if (mark) {
          // corner triangles: top-right = question term, bottom-right = answer term
          if (mark === "q" || mark === "qa") {
            ctx.fillStyle = "#9b2f5f";
            ctx.beginPath();
            ctx.moveTo(x + cellW - 6, y + 1);
            ctx.lineTo(x + cellW - 1, y + 1);
            ctx.lineTo(x + cellW - 1, y + 6);
            ctx.fill();
          }
          if (mark === "a" || mark === "qa") {
            ctx.fillStyle = "#caa24a";
            ctx.beginPath();
            ctx.moveTo(x + cellW - 1, y + cellH - 6);
            ctx.lineTo(x + cellW - 1, y + cellH - 1);
            ctx.lineTo(x + cellW - 6, y + cellH - 1);
            ctx.fill();
          }
        }
      }
      ctx.globalAlpha = 1;
    }

    if (highlightRow != null && highlightRow >= 0 && highlightRow < nRows) {
      ctx.strokeStyle = "#113f8c";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, headerH + highlightRow * cellH + 0.5, width - 1, cellH - 1);
    }
    if (highlightCol != null && highlightCol >= 0 && highlightCol < nCols) {
      ctx.strokeStyle = "#113f8c";
      ctx.lineWidth = 1;
      ctx.strokeRect(labelW + highlightCol * cellW + 0.5, headerH + 0.5, cellW - 1, height - headerH - 1);
    }
    if (selected) {
      ctx.strokeStyle = "#9b2f5f";
      ctx.lineWidth = 2;
      ctx.strokeRect(
        labelW + selected.c * cellW + 1,
        headerH + selected.r * cellH + 1,
        cellW - 2,
        cellH - 2,
      );
    }
  }, [rowLabels, colLabels, values, selected, highlightRow, highlightCol, revealUpToRow, revealUpToCol, marks, width, height, cellW, cellH, labelW, maxAbs, nRows, nCols]);

  return (
    <div className="canvas-scroll">
      <canvas
        ref={ref}
        style={{ cursor: "pointer" }}
        onClick={(e) => {
          const rect = ref.current!.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          if (x < labelW || y < headerH) return;
          const c = Math.floor((x - labelW) / cellW);
          const r = Math.floor((y - headerH) / cellH);
          if (r >= 0 && r < nRows && c >= 0 && c < nCols) onPick(r, c);
        }}
      />
    </div>
  );
}
