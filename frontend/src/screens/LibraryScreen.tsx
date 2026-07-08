import { useEffect, useState } from "react";
import { getLibrary } from "../api/client";
import type { LibraryItem } from "../trace/types";
import { HonestyBanner } from "../components/HonestyBanner";

interface Props {
  onOpen(traceId: string): void;
  onUpload(): void;
}

/** Library screen: previously computed traces, re-openable instantly. */
export function LibraryScreen({ onOpen, onUpload }: Props) {
  const [items, setItems] = useState<LibraryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getLibrary()
      .then(setItems)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="app">
      <div className="topbar" style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h1>Video J-Space Replay</h1>
        <button className="btn primary" onClick={onUpload}>+ new</button>
      </div>
      <div style={{ padding: "10px 12px 0" }}>
        <HonestyBanner />
      </div>
      <div className="center-screen">
        <div className="uctitle" style={{ fontSize: 12, letterSpacing: "0.14em" }}>
          Library
        </div>
        {error && <div className="err">{error}</div>}
        {items && items.length === 0 && <div className="muted">no traces yet.</div>}
        {(items ?? []).map((it) => (
          <div key={it.trace_id} className="lib-item" onClick={() => onOpen(it.trace_id)}>
            <div style={{ minWidth: 0 }}>
              <div className="lib-q">{it.question}</div>
              <div className="lib-a">{it.answer}</div>
            </div>
            <div className="concept-meta">
              {it.duration_s}s · {it.trace_id}
            </div>
          </div>
        ))}
        {!items && !error && <div className="muted">loading…</div>}
      </div>
    </div>
  );
}
