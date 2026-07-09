import { useEffect, useRef, useState } from "react";
import { getLenses, isCached, startTrace, uploadVideo } from "../api/client";
import { DEFAULT_QUESTION } from "../constants";
import { HonestyBanner } from "../components/HonestyBanner";
import { LensSelect } from "../components/QueryConsole";

interface Props {
  onJobStarted(jobId: string, question: string, videoId: string): void;
  onCached(traceId: string): void;
  onOpenLibrary(): void;
}

/** Upload → question form (default prefilled) → kicks off the pipeline. */
export function UploadScreen({ onJobStarted, onCached, onOpenLibrary }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lenses, setLenses] = useState<string[]>([]);
  const [lens, setLens] = useState("logit-lens-v1");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let alive = true;
    getLenses().then((info) => {
      if (!alive) return;
      setLenses(info.lenses);
      // prefer the J-lens when this install has one fitted (issue #8: it is
      // the paper's method; the trace is badged either way)
      setLens(info.lenses.includes("j-lens-v1") ? "j-lens-v1" : info.default);
    });
    return () => {
      alive = false;
    };
  }, []);

  const submit = async () => {
    if (!file || busy) return;
    setBusy(true);
    setError(null);
    try {
      const up = await uploadVideo(file);
      const res = await startTrace(up.video_id, question.trim() || DEFAULT_QUESTION, lens);
      if (isCached(res)) onCached(res.trace_id);
      else onJobStarted(res.job_id, question.trim() || DEFAULT_QUESTION, up.video_id);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  };

  return (
    <div className="app">
      <div className="topbar" style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h1>Video J-Space Replay</h1>
        <button className="btn" onClick={onOpenLibrary}>library</button>
      </div>
      <div style={{ padding: "10px 12px 0" }}>
        <HonestyBanner />
      </div>
      <div className="center-screen">
        <div className="field">
          <label>1 · video clip (5–20s)</label>
          <div
            className={"dropzone" + (drag ? " drag" : "")}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
            }}
          >
            {file ? (
              <span>
                selected: <b>{file.name}</b> ({Math.round(file.size / 1024)} KB)
              </span>
            ) : (
              <span>drop a video here, or click to choose</span>
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            style={{ display: "none" }}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <div className="axis-note">
            dev mode accepts any file — it maps to a committed fixture trace.
          </div>
        </div>

        <div className="field">
          <label>2 · question</label>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            aria-label="question"
          />
        </div>

        {lenses.length > 1 && (
          <div className="field">
            <label>3 · decode lens</label>
            <LensSelect value={lens} onChange={setLens} lenses={lenses} />
            <div className="axis-note">
              J-lens: analytic Jacobian transport (issue #8) — richer mid/late-layer
              readouts; logit lens: the raw identity readout. The replay is badged
              with whichever produced it.
            </div>
          </div>
        )}

        {error && <div className="err">{error}</div>}

        <div>
          <button className="btn primary" disabled={!file || busy} onClick={submit}>
            {busy ? "starting…" : "run replay ▶"}
          </button>
        </div>
      </div>
    </div>
  );
}
