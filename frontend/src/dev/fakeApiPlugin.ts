/**
 * Vite dev-server middleware that fakes the M3 backend API contract from the
 * committed fixture traces, so the whole frontend loop runs with ZERO backend
 * and no GPU. Registered from vite.config.ts. Not part of the production build.
 *
 * Contract faked (see issue #6):
 *   POST /videos            (multipart "file") -> {video_id, filename, duration_s}
 *   POST /traces            {video_id, question?} -> 202 {job_id, queue_position}
 *                                                    | 200 {trace_id, cached:true}
 *   GET  /jobs/{job_id}     -> job status, walking the stages with short delays
 *   GET  /traces/{trace_id} -> trace JSON
 *   GET  /library          -> {items: [...]}
 *   GET  /videos/{id}/file -> raw video (fixtures/clips/<id>.<ext>) or 404
 *
 * Everything is keyed by the fixture base name (e.g. "ball_drop"), used as both
 * video_id and trace_id, so library items are directly re-openable.
 */
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Connect, Plugin } from "vite";
import type { ServerResponse } from "node:http";

const STAGES = [
  "sampling",
  "prefill_capture",
  "generating",
  "lens_decode",
  "labels",
  "grounding",
  "done",
] as const;

const STAGE_MS = 650; // per stage; whole fake run ~4.5s
const QUEUE_MS = 500; // time spent "queued" before "running"

const VIDEO_EXTS = ["mp4", "webm", "mov", "m4v", "ogg"];

interface Job {
  jobId: string;
  traceName: string;
  question: string;
  startedAt: number;
}

function baseNames(tracesDir: string): string[] {
  if (!existsSync(tracesDir)) return [];
  return readdirSync(tracesDir)
    .filter((f) => f.endsWith(".trace.json"))
    .map((f) => f.replace(/\.trace\.json$/, ""))
    .sort();
}

function readTrace(tracesDir: string, name: string): unknown | null {
  const p = resolve(tracesDir, `${name}.trace.json`);
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf8"));
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const s = JSON.stringify(body);
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.setHeader("Cache-Control", "no-store");
  res.end(s);
}

function drain(req: Connect.IncomingMessage): Promise<void> {
  return new Promise((res) => {
    req.on("data", () => {});
    req.on("end", () => res());
    req.on("error", () => res());
  });
}

export function fakeApiPlugin(): Plugin {
  const here = dirname(fileURLToPath(import.meta.url));
  // src/dev -> frontend -> repo root
  const repoRoot = resolve(here, "..", "..", "..");
  const tracesDir = resolve(repoRoot, "fixtures", "traces");
  const clipsDir = resolve(repoRoot, "fixtures", "clips");

  const jobs = new Map<string, Job>();
  // cache key `${video}::${question}` -> traceName, to exercise the 200 cached path
  const cache = new Map<string, string>();
  let jobSeq = 0;
  let videoSeq = 0;
  // video_id -> fixture base name (uploads round-robin over available fixtures)
  const videoToName = new Map<string, string>();

  function traceDuration(name: string): number {
    const t = readTrace(tracesDir, name) as
      | { frame_groups?: Array<{ time_end: number }> }
      | null;
    const groups = t?.frame_groups ?? [];
    return groups.length ? groups[groups.length - 1].time_end : 0;
  }

  function resolveTraceName(id: string): string | null {
    const names = baseNames(tracesDir);
    if (names.includes(id)) return id;
    // uploaded video_id mapped to a fixture
    return videoToName.get(id) ?? null;
  }

  return {
    name: "jsr-fake-api",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = (req.url ?? "").split("?")[0];
        const method = (req.method ?? "GET").toUpperCase();
        const names = baseNames(tracesDir);

        // POST /videos
        if (method === "POST" && url === "/videos") {
          await drain(req);
          const name = names.length ? names[videoSeq % names.length] : "unknown";
          videoSeq += 1;
          const videoId = `${name}`; // key by fixture base name
          videoToName.set(videoId, name);
          sendJson(res, 200, {
            video_id: videoId,
            filename: `${name}.mp4`,
            duration_s: traceDuration(name),
          });
          return;
        }

        // GET /lenses — dev fake offers both so the pickers render; the
        // fixture traces themselves are logit-lens either way
        if (method === "GET" && url === "/lenses") {
          sendJson(res, 200, {
            lenses: ["logit-lens-v1", "j-lens-v1"],
            default: "logit-lens-v1",
          });
          return;
        }

        // POST /traces
        if (method === "POST" && url === "/traces") {
          const chunks: Buffer[] = [];
          req.on("data", (c: Buffer) => chunks.push(c));
          await new Promise<void>((r) => req.on("end", () => r()));
          let body: { video_id?: string; question?: string; lens?: string } = {};
          try {
            body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
          } catch {
            body = {};
          }
          const videoId = body.video_id ?? (names[0] ?? "unknown");
          const question = body.question ?? "Describe what happens in this video.";
          const traceName = videoToName.get(videoId) ?? (names.includes(videoId) ? videoId : names[0]);
          const key = `${videoId}::${question}::${body.lens ?? "logit-lens-v1"}`;
          if (cache.has(key)) {
            sendJson(res, 200, { trace_id: cache.get(key), cached: true });
            return;
          }
          const jobId = `job_${++jobSeq}`;
          jobs.set(jobId, { jobId, traceName, question, startedAt: Date.now() });
          cache.set(key, traceName);
          sendJson(res, 202, { job_id: jobId, queue_position: 0 });
          return;
        }

        // GET /jobs/{job_id}
        const jobMatch = url.match(/^\/jobs\/([^/]+)$/);
        if (method === "GET" && jobMatch) {
          const job = jobs.get(jobMatch[1]);
          if (!job) {
            sendJson(res, 404, { error: "unknown job" });
            return;
          }
          const elapsed = Date.now() - job.startedAt;
          if (elapsed < QUEUE_MS) {
            sendJson(res, 200, {
              job_id: job.jobId,
              status: "queued",
              queue_position: 0,
              stages_done: [],
            });
            return;
          }
          const stageIdx = Math.min(
            STAGES.length - 1,
            Math.floor((elapsed - QUEUE_MS) / STAGE_MS),
          );
          const stage = STAGES[stageIdx];
          const stagesDone = STAGES.slice(0, stageIdx).filter((s) => s !== "done");
          if (stage === "done") {
            sendJson(res, 200, {
              job_id: job.jobId,
              status: "done",
              stage: "done",
              stages_done: STAGES.slice(0, STAGES.length - 1),
              trace_id: job.traceName,
              warning:
                "Demo fixture trace (no GPU run). Concepts/grounding may be empty.",
            });
            return;
          }
          sendJson(res, 200, {
            job_id: job.jobId,
            status: "running",
            stage,
            stages_done: stagesDone,
          });
          return;
        }

        // GET /traces/{trace_id}
        const traceMatch = url.match(/^\/traces\/([^/]+)$/);
        if (method === "GET" && traceMatch) {
          const name = resolveTraceName(traceMatch[1]);
          const trace = name ? readTrace(tracesDir, name) : null;
          if (!trace) {
            sendJson(res, 404, { error: "unknown trace" });
            return;
          }
          sendJson(res, 200, trace);
          return;
        }

        // GET /library
        if (method === "GET" && url === "/library") {
          const items = names.map((name) => {
            const t = readTrace(tracesDir, name) as {
              video_id?: string;
              question?: string;
              answer?: string;
              frame_groups?: Array<{ time_end: number }>;
            } | null;
            return {
              trace_id: name,
              video_id: name,
              question: t?.question ?? "",
              answer: t?.answer ?? "",
              created_at: new Date().toISOString(),
              duration_s: traceDuration(name),
            };
          });
          sendJson(res, 200, { items });
          return;
        }

        // GET /videos/{id}/file
        const fileMatch = url.match(/^\/videos\/([^/]+)\/file$/);
        if (method === "GET" && fileMatch) {
          const name = resolveTraceName(fileMatch[1]) ?? fileMatch[1];
          let served = false;
          for (const ext of VIDEO_EXTS) {
            const p = resolve(clipsDir, `${name}.${ext}`);
            if (existsSync(p)) {
              const data = readFileSync(p);
              res.statusCode = 200;
              res.setHeader(
                "Content-Type",
                ext === "webm" ? "video/webm" : "video/mp4",
              );
              res.setHeader("Content-Length", String(data.length));
              res.end(data);
              served = true;
              break;
            }
          }
          if (!served) {
            // Fixture clips are gitignored; the player handles this 404 with a
            // "video unavailable — timeline still scrubbable" state.
            sendJson(res, 404, { error: "video unavailable" });
          }
          return;
        }

        next();
      });
    },
  };
}
