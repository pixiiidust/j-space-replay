import { useEffect, useState } from "react";
import { pollJob, type JobState } from "../api/client";
import { PIPELINE_STAGES } from "../constants";
import { HonestyBanner } from "../components/HonestyBanner";

interface Props {
  jobId: string;
  question: string;
  onDone(traceId: string): void;
}

/** Job progress → auto-transition to replay when the trace is ready. */
export function ProgressScreen({ jobId, question, onDone }: Props) {
  const [state, setState] = useState<JobState | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    pollJob(jobId, (s) => alive && setState(s))
      .then((final) => {
        if (!alive) return;
        if (final.status === "done" && final.trace_id) onDone(final.trace_id);
        else if (final.status === "error") setError(final.error ?? "pipeline error");
      })
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, [jobId, onDone]);

  const done = new Set(state?.stages_done ?? []);
  const current = state?.stage;

  return (
    <div className="app">
      <div className="topbar">
        <h1>Video J-Space Replay</h1>
        <div className="metaline">
          processing · one offline pass · prompt: <b>{question}</b>
        </div>
      </div>
      <div style={{ padding: "10px 12px 0" }}>
        <HonestyBanner />
      </div>
      <div className="center-screen">
        <div className="panel">
          <div className="panel-h">
            <span>Pipeline</span>
            <span className="muted">{state?.status ?? "starting"}</span>
          </div>
          <div className="panel-b">
            {state?.status === "queued" && (
              <div className="muted" style={{ marginBottom: 6 }}>
                queued · position {state.queue_position ?? 0}
              </div>
            )}
            <ul className="stages">
              {PIPELINE_STAGES.filter((s) => s !== "done").map((s) => {
                const isDone = done.has(s);
                const isActive = current === s;
                const cls = isDone ? "done" : isActive ? "active" : "pending";
                return (
                  <li key={s} className={cls}>
                    <span className="dot" />
                    <span className="name">{s}</span>
                    <span className="muted" style={{ marginLeft: "auto", fontSize: 10 }}>
                      {isDone ? "done" : isActive ? "running…" : "pending"}
                    </span>
                  </li>
                );
              })}
            </ul>
            {state?.warning && (
              <div className="axis-note warn" style={{ marginTop: 8 }}>
                {state.warning}
              </div>
            )}
            {error && <div className="err" style={{ marginTop: 8 }}>{error}</div>}
          </div>
        </div>
        <div className="axis-note">
          Process first, replay second — never computed live during playback.
        </div>
      </div>
    </div>
  );
}
