import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import type { Grounding } from "../trace/types";
import { setupCanvas } from "./canvas";

export type OverlayMode = "clean" | "boxes" | "patch";

interface PatchOverlay {
  rows: number;
  cols: number;
  values: number[];
  label: string;
}

interface Props {
  videoRef: RefObject<HTMLVideoElement | null>;
  src: string;
  videoAvailable: boolean;
  onUnavailable(): void;
  mode: OverlayMode;
  boxes: Grounding[];
  patch: PatchOverlay | null;
}

/** Contained rect of a natural-size frame inside a letterboxed container. */
function containRect(cw: number, ch: number, nw: number, nh: number) {
  if (nw <= 0 || nh <= 0) return { x: 0, y: 0, w: cw, h: ch };
  const scale = Math.min(cw / nw, ch / nh);
  const w = nw * scale;
  const h = nh * scale;
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
}

/**
 * Video panel: the clip (via GET /videos/{id}/file) or, when the clip is
 * unavailable (fixture clips are gitignored), a "video unavailable — timeline
 * still scrubbable" placeholder. Overlays: grounding boxes / patch heatmap for
 * the selected concept, drawn on a canvas above the frame.
 */
export function VideoPanel({
  videoRef,
  src,
  videoAvailable,
  onUnavailable,
  mode,
  boxes,
  patch,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [nat, setNat] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = overlayRef.current;
    if (!wrap || !canvas) return;
    const cw = wrap.clientWidth;
    const ch = wrap.clientHeight;
    const ctx = setupCanvas(canvas, cw, ch);
    if (!ctx) return;
    ctx.clearRect(0, 0, cw, ch);
    if (mode === "clean") return;

    const rect = videoAvailable && nat.w > 0 ? containRect(cw, ch, nat.w, nat.h) : { x: 0, y: 0, w: cw, h: ch };

    if (mode === "patch" && patch) {
      const cellW = rect.w / patch.cols;
      const cellH = rect.h / patch.rows;
      for (let i = 0; i < patch.values.length; i++) {
        const v = patch.values[i];
        if (!v) continue;
        const r = Math.floor(i / patch.cols);
        const c = i % patch.cols;
        ctx.fillStyle = "rgba(17,63,140,0.42)";
        ctx.fillRect(rect.x + c * cellW, rect.y + r * cellH, cellW, cellH);
        ctx.strokeStyle = "rgba(17,63,140,0.7)";
        ctx.lineWidth = 1;
        ctx.strokeRect(rect.x + c * cellW, rect.y + r * cellH, cellW, cellH);
      }
    }

    if (mode === "boxes") {
      if (!videoAvailable || nat.w === 0) {
        ctx.fillStyle = "#e8a0bf";
        ctx.font = "11px 'IBM Plex Mono', monospace";
        ctx.fillText("grounding overlay needs the video frame", 10, 18);
      }
      ctx.strokeStyle = "#9b2f5f";
      ctx.lineWidth = 2;
      ctx.font = "11px 'IBM Plex Mono', monospace";
      for (const b of boxes) {
        const sx = rect.w / (nat.w || rect.w);
        const sy = rect.h / (nat.h || rect.h);
        const x = rect.x + b.box[0] * sx;
        const y = rect.y + b.box[1] * sy;
        const w = (b.box[2] - b.box[0]) * sx;
        const h = (b.box[3] - b.box[1]) * sy;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = "#9b2f5f";
        ctx.fillText(b.label, x + 2, y - 4);
      }
    }
  }, [mode, boxes, patch, nat, videoAvailable]);

  return (
    <div className="video-wrap" ref={wrapRef}>
      {videoAvailable ? (
        <video
          ref={videoRef}
          src={src}
          onLoadedMetadata={(e) => {
            const v = e.currentTarget;
            setNat({ w: v.videoWidth, h: v.videoHeight });
          }}
          onError={onUnavailable}
          playsInline
          muted
        />
      ) : (
        <div className="video-missing">
          <div className="big">video unavailable</div>
          timeline still scrubbable
          <br />
          (fixture clips are gitignored — regenerate with
          <br />
          <code>uv run python scripts/make_fixtures.py</code>)
        </div>
      )}
      <canvas ref={overlayRef} className="video-overlay" />
    </div>
  );
}
