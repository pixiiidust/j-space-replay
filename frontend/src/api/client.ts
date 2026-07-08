/**
 * API client for the M3 backend contract (issue #6). In dev these hit the Vite
 * fake-api middleware; in prod they hit the FastAPI backend on the same origin.
 * Base URL is configurable via VITE_API_BASE (default: same origin).
 */
import type { LibraryItem, Trace } from "../trace/types";

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export interface UploadResult {
  video_id: string;
  filename: string;
  duration_s: number;
}

export interface StartTraceQueued {
  job_id: string;
  queue_position: number;
}
export interface StartTraceCached {
  trace_id: string;
  cached: true;
}
export type StartTraceResult = StartTraceQueued | StartTraceCached;

export type JobStatus = "queued" | "running" | "done" | "error";

export interface JobState {
  job_id: string;
  status: JobStatus;
  queue_position?: number;
  stage?: string;
  stages_done: string[];
  trace_id?: string;
  error?: string;
  warning?: string;
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = "";
    try {
      detail = JSON.stringify(await res.json());
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status} ${res.statusText} ${detail}`.trim());
  }
  return (await res.json()) as T;
}

export async function uploadVideo(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/videos`, { method: "POST", body: form });
  return jsonOrThrow<UploadResult>(res);
}

export async function startTrace(
  videoId: string,
  question?: string,
): Promise<StartTraceResult> {
  const res = await fetch(`${BASE}/traces`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId, question }),
  });
  // 202 -> queued job; 200 -> cached trace
  return jsonOrThrow<StartTraceResult>(res);
}

export function isCached(r: StartTraceResult): r is StartTraceCached {
  return (r as StartTraceCached).cached === true;
}

export async function getJob(jobId: string): Promise<JobState> {
  const res = await fetch(`${BASE}/jobs/${encodeURIComponent(jobId)}`);
  return jsonOrThrow<JobState>(res);
}

export async function getTrace(traceId: string): Promise<Trace> {
  const res = await fetch(`${BASE}/traces/${encodeURIComponent(traceId)}`);
  return jsonOrThrow<Trace>(res);
}

export async function getLibrary(): Promise<LibraryItem[]> {
  const res = await fetch(`${BASE}/library`);
  const data = await jsonOrThrow<{ items: LibraryItem[] }>(res);
  return data.items;
}

export function videoFileUrl(videoId: string): string {
  return `${BASE}/videos/${encodeURIComponent(videoId)}/file`;
}

/** Poll a job to completion, invoking `onUpdate` on each state. */
export async function pollJob(
  jobId: string,
  onUpdate: (s: JobState) => void,
  intervalMs = 400,
): Promise<JobState> {
  for (;;) {
    const s = await getJob(jobId);
    onUpdate(s);
    if (s.status === "done" || s.status === "error") return s;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
