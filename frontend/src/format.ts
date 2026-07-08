/** Small display formatters. */

/** mm:ss.mmm-ish -> "00:04.50" (minutes:seconds.hundredths), per SPEC HUD. */
export function fmtClock(t: number): string {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  const cs = Math.floor((t - Math.floor(t)) * 100);
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(m)}:${pad(s)}.${pad(cs)}`;
}

export function fmtStrength(v: number): string {
  return v.toFixed(2);
}
