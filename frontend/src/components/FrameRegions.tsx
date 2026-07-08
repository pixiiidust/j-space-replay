import { useEffect, useRef } from "react";
import type { Grounding } from "../trace/types";
import type { OverlayMode } from "./VideoPanel";
import { setupCanvas } from "./canvas";

interface PatchData {
  rows: number;
  cols: number;
  values: number[];
  label: string;
  layer: number | null;
}

interface Props {
  mode: OverlayMode;
  onModeChange(m: OverlayMode): void;
  patch: PatchData | null;
  grounding: Grounding[];
}

/**
 * Frame regions panel: overlay mode selector (clean / grounding boxes / patch),
 * a standalone patch heatmap for the selected concept (renders without the
 * video), and the grounding boxes list for the current time.
 */
export function FrameRegions({ mode, onModeChange, patch, grounding }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const cssW = 240;
  const cssH = patch ? Math.round((cssW / patch.cols) * patch.rows) : 80;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = setupCanvas(canvas, cssW, cssH);
    if (!ctx) return;
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, cssW, cssH);
    if (!patch) {
      ctx.fillStyle = "#8b8b84";
      ctx.font = "10px 'IBM Plex Mono', monospace";
      ctx.fillText("select a concept", 8, 16);
      return;
    }
    const cw = cssW / patch.cols;
    const chh = cssH / patch.rows;
    for (let i = 0; i < patch.values.length; i++) {
      const r = Math.floor(i / patch.cols);
      const c = i % patch.cols;
      const v = patch.values[i];
      ctx.fillStyle = v ? "rgba(17,63,140,0.85)" : "#1b1b1b";
      ctx.fillRect(c * cw, r * chh, cw - 0.5, chh - 0.5);
    }
  }, [patch, cssH]);

  const modes: OverlayMode[] = ["clean", "boxes", "patch"];

  return (
    <div className="panel">
      <div className="panel-h">
        <span>Frame Regions</span>
        <div className="mini-actions">
          {modes.map((m) => (
            <button key={m} className={mode === m ? "" : ""} style={mode === m ? { borderColor: "#113f8c", color: "#113f8c" } : undefined} onClick={() => onModeChange(m)}>
              {m}
            </button>
          ))}
        </div>
      </div>
      <div className="panel-b">
        <div className="axis-note" style={{ marginBottom: 4 }}>
          patch heatmap{patch ? `: "${patch.label}" · layer ${patch.layer ?? "n/a"}` : " (none selected)"}
        </div>
        <div className="canvas-scroll">
          <canvas ref={ref} />
        </div>
        <div className="axis-note" style={{ marginTop: 6 }}>
          grounding boxes {grounding.length ? `(${grounding.length})` : "— none in this trace"}
        </div>
        {grounding.map((b, i) => (
          <div key={i} className="concept-meta" style={{ padding: "1px 0" }}>
            {b.label} [{b.box.join(", ")}] t={b.time}s
          </div>
        ))}
      </div>
    </div>
  );
}
