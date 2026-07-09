import { useCallback, useState } from "react";
import { startTrace, isCached } from "./api/client";
import { UploadScreen } from "./screens/UploadScreen";
import { ProgressScreen } from "./screens/ProgressScreen";
import { ReplayScreen } from "./screens/ReplayScreen";
import { LibraryScreen } from "./screens/LibraryScreen";

type Route =
  | { screen: "upload" }
  | { screen: "progress"; jobId: string; question: string }
  | { screen: "replay"; traceId: string }
  | { screen: "library" };

/**
 * Top-level screen state machine driving the full loop:
 * upload → progress → replay → (scrub/drill) → re-ask → progress → replay,
 * plus the library screen. No dev-tools knowledge required.
 */
export function App() {
  const [route, setRoute] = useState<Route>({ screen: "upload" });

  const openReplay = useCallback((traceId: string) => setRoute({ screen: "replay", traceId }), []);

  // "ask another question" from the replay screen re-runs the pipeline.
  const reAsk = useCallback(async (videoId: string, question: string, lens?: string) => {
    const res = await startTrace(videoId, question, lens);
    if (isCached(res)) setRoute({ screen: "replay", traceId: res.trace_id });
    else setRoute({ screen: "progress", jobId: res.job_id, question });
  }, []);

  switch (route.screen) {
    case "upload":
      return (
        <UploadScreen
          onJobStarted={(jobId, question) => setRoute({ screen: "progress", jobId, question })}
          onCached={openReplay}
          onOpenLibrary={() => setRoute({ screen: "library" })}
        />
      );
    case "progress":
      return (
        <ProgressScreen jobId={route.jobId} question={route.question} onDone={openReplay} />
      );
    case "replay":
      return (
        <ReplayScreen
          traceId={route.traceId}
          onReAsk={reAsk}
          onOpenLibrary={() => setRoute({ screen: "library" })}
        />
      );
    case "library":
      return (
        <LibraryScreen onOpen={openReplay} onUpload={() => setRoute({ screen: "upload" })} />
      );
  }
}
