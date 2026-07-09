import { useEffect, useMemo, useRef, useState } from "react";
import type { Trace } from "../trace/types";
import { getTrace, videoFileUrl } from "../api/client";
import { buildTimeline, traceDuration } from "../trace/selectors";
import { deriveEvents } from "../trace/events";
import { bestLayerForWord, patchHeatmapForWord } from "../trace/jspace";
import { useClock } from "../hooks/useClock";
import { HonestyBanner } from "../components/HonestyBanner";
import { VideoPanel, type OverlayMode } from "../components/VideoPanel";
import { Controls } from "../components/Controls";
import { QueryConsole } from "../components/QueryConsole";
import { AnswerWorkspace, LensChip } from "../components/JSpaceSlice";
import { ConceptBoard } from "../components/ConceptBoard";
import { EventLog } from "../components/EventLog";

interface Props {
  traceId: string;
  onReAsk(videoId: string, question: string, lens?: string): void;
  onOpenLibrary(): void;
}

export function ReplayScreen({ traceId, onReAsk, onOpenLibrary }: Props) {
  const [trace, setTrace] = useState<Trace | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [videoAvailable, setVideoAvailable] = useState(true);
  const [selectedRow, setSelectedRow] = useState<number>(0);
  const [pinned, setPinned] = useState<Set<string>>(new Set());
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [overlay, setOverlay] = useState<OverlayMode>("clean");
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let alive = true;
    setTrace(null);
    setError(null);
    setVideoAvailable(true);
    setSelectedRow(0);
    getTrace(traceId)
      .then((t) => {
        if (alive) setTrace(t);
      })
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, [traceId]);

  if (error) return <div className="center-screen"><div className="err">failed to load trace: {error}</div></div>;
  if (!trace) return <div className="center-screen muted">loading trace…</div>;

  return <ReplayBody
    trace={trace}
    videoRef={videoRef}
    videoAvailable={videoAvailable}
    setVideoAvailable={setVideoAvailable}
    selectedRow={selectedRow}
    setSelectedRow={setSelectedRow}
    pinned={pinned}
    setPinned={setPinned}
    hidden={hidden}
    setHidden={setHidden}
    overlay={overlay}
    setOverlay={setOverlay}
    onReAsk={onReAsk}
    onOpenLibrary={onOpenLibrary}
  />;
}

// Split so hooks below run only once the trace is loaded.
function ReplayBody(props: {
  trace: Trace;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  videoAvailable: boolean;
  setVideoAvailable(b: boolean): void;
  selectedRow: number;
  setSelectedRow(n: number): void;
  pinned: Set<string>;
  setPinned(s: Set<string>): void;
  hidden: Set<string>;
  setHidden(s: Set<string>): void;
  overlay: OverlayMode;
  setOverlay(m: OverlayMode): void;
  onReAsk(videoId: string, question: string, lens?: string): void;
  onOpenLibrary(): void;
}) {
  const {
    trace, videoRef, videoAvailable, setVideoAvailable, selectedRow, setSelectedRow,
    pinned, setPinned, hidden, setHidden, overlay, setOverlay, onReAsk, onOpenLibrary,
  } = props;

  const model = useMemo(() => buildTimeline(trace), [trace]);
  const events = useMemo(() => deriveEvents(model), [model]);
  const clock = useClock(trace, videoRef, videoAvailable);

  const selRow = model.rows[selectedRow] ?? model.rows[0] ?? null;
  const curGroup = trace.frame_groups[clock.groupIndex];

  // which answer token the replay is currently "forming"
  const nAnswer = trace.answer_tokens.length;
  const fraction = clock.duration > 0 ? Math.min(1, clock.time / clock.duration) : 1;
  const answerRow = nAnswer > 0 ? Math.min(nAnswer - 1, Math.floor(fraction * nAnswer)) : 0;

  // patch heatmap for the selected concept at the current frame group
  const patch = useMemo(() => {
    if (!selRow || !curGroup) return null;
    const word = selRow.sourceTokens[0] ?? selRow.label;
    const layer = bestLayerForWord(curGroup, word);
    if (layer == null) return null;
    const hm = patchHeatmapForWord(curGroup, layer, trace.meta.token_strings, word);
    if (!hm) return null;
    return { ...hm, label: selRow.label, layer };
  }, [selRow, curGroup, trace.meta.token_strings]);

  const groundingNow = trace.grounding.filter(
    (b) => curGroup && b.time >= curGroup.time_start && b.time < curGroup.time_end,
  );

  const toggleSet = (s: Set<string>, setS: (x: Set<string>) => void, label: string) => {
    const next = new Set(s);
    if (next.has(label)) next.delete(label);
    else next.add(label);
    setS(next);
  };

  const dur = traceDuration(trace);
  const meta = trace.meta;

  return (
    <div className="app">
      <div className="topbar">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
          <h1>Video J-Space Replay</h1>
          <div className="controls">
            <button className="btn" onClick={onOpenLibrary}>library</button>
          </div>
        </div>
        <div className="metaline">
          clip <b>{dur}s</b> | frame groups <b>{trace.frame_groups.length}</b> | layers{" "}
          <b>{meta.n_layers}</b> | concepts <b>{model.mode === "concepts" ? model.rows.length : 0}</b>{" "}
          | model <b>{meta.model}</b> | <LensChip lens={meta.lens} caveats={meta.lens_caveats} />
        </div>
      </div>

      <div style={{ padding: "8px 12px 0" }}>
        <HonestyBanner />
      </div>

      {/* question in, answer out — the contract; the machinery below explains it */}
      <div style={{ padding: "8px 12px 0" }}>
        <QueryConsole
          trace={trace}
          answerRow={answerRow}
          onSeekToken={(i) =>
            clock.seek(nAnswer > 0 ? ((i + 0.5) / nAnswer) * clock.duration : 0)
          }
          onReAsk={onReAsk}
        />
      </div>

      <div className="dash">
        {/* LEFT — the clip and its transport */}
        <div className="col">
          <div className="panel">
            <div className="panel-h">
              <span>Video Player</span>
              <span className="controls" style={{ gap: 4 }}>
                {(["clean", "boxes", "patch"] as OverlayMode[]).map((m) => (
                  <button
                    key={m}
                    className={"btn" + (overlay === m ? " active" : "")}
                    onClick={() => setOverlay(m)}
                  >
                    {m}
                  </button>
                ))}
              </span>
            </div>
            <VideoPanel
              videoRef={videoRef}
              src={videoFileUrl(trace.video_id)}
              videoAvailable={videoAvailable}
              onUnavailable={() => setVideoAvailable(false)}
              mode={overlay}
              boxes={groundingNow}
              patch={patch}
            />
          </div>
          <Controls clock={clock} trace={trace} model={model} />
        </div>

        {/* CENTER — hero: the answer forming across layers on the playback clock */}
        <div className="col">
          <AnswerWorkspace trace={trace} answerRow={answerRow} playing={clock.playing} />
        </div>

        {/* RIGHT — sidebar: derived views */}
        <div className="col">
          <ConceptBoard
            model={model}
            selectedRow={selectedRow}
            pinned={pinned}
            hidden={hidden}
            onSelect={setSelectedRow}
            onJumpPeak={(idx) => {
              setSelectedRow(idx);
              clock.seek(model.rows[idx].peakTime);
            }}
            onTogglePin={(l) => toggleSet(pinned, setPinned, l)}
            onToggleHide={(l) => toggleSet(hidden, setHidden, l)}
          />
          <EventLog
            events={events}
            onPick={(ev) => {
              clock.seek(ev.time);
              const idx = model.rows.findIndex((r) => r.label === ev.label);
              if (idx >= 0) setSelectedRow(idx);
            }}
          />
        </div>
      </div>
    </div>
  );
}
