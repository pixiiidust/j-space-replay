/** Canvas helpers shared by the heatmap/grid renderers. */

/** Size a canvas for crisp rendering at devicePixelRatio and return a 2D ctx
 *  scaled to CSS pixels. */
export function setupCanvas(
  canvas: HTMLCanvasElement,
  cssW: number,
  cssH: number,
): CanvasRenderingContext2D | null {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/** Navy heat ramp: 0 -> near-panel grey, 1 -> primary navy. No gradient fills,
 *  just a per-cell solid colour so the grid stays crisp. */
export function heatColor(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  // from #f1f1ec (241,241,236) to #113f8c (17,63,140)
  const r = Math.round(241 + (17 - 241) * c);
  const g = Math.round(241 + (63 - 241) * c);
  const b = Math.round(236 + (140 - 236) * c);
  return `rgb(${r},${g},${b})`;
}

/** Text colour that stays legible over a heat cell of intensity t. */
export function heatTextColor(t: number): string {
  return t > 0.55 ? "#ffffff" : "#1d1d1b";
}
