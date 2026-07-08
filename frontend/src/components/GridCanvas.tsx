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

    ctx.fillStyle = "#8b8b84";
    ctx.textAlign = "center";
    for (let c = 0; c < nCols; c++) {
      ctx.fillText(colLabels[c], labelW + c * cellW + cellW / 2, headerH / 2);
    }

    for (let r = 0; r < nRows; r++) {
      const y = headerH + r * cellH;
      ctx.textAlign = "left";
      ctx.fillStyle = highlightRow === r ? "#113f8c" : "#1d1d1b";
      const lbl = rowLabels[r].length > 13 ? rowLabels[r].slice(0, 12) + "…" : rowLabels[r];
      ctx.fillText(lbl, 4, y + cellH / 2);
      for (let c = 0; c < nCols; c++) {
        const v = values[r][c];
        const x = labelW + c * cellW;
        ctx.fillStyle = v == null ? "#f7f7f5" : heatColor(Math.abs(v) / maxAbs);
        ctx.fillRect(x + 0.5, y + 0.5, cellW - 1, cellH - 1);
      }
    }

    if (highlightRow != null && highlightRow >= 0 && highlightRow < nRows) {
      ctx.strokeStyle = "#113f8c";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, headerH + highlightRow * cellH + 0.5, width - 1, cellH - 1);
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
  }, [rowLabels, colLabels, values, selected, highlightRow, width, height, cellW, cellH, labelW, maxAbs, nRows, nCols]);

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
