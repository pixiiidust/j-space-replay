"""FastAPI surface for J-Space-Replay — the exact contract the M4 frontend builds on.

    uv run uvicorn jsr.server.api:app

Routes:
  POST /videos                (multipart "file")  -> {video_id, filename, duration_s}
  POST /traces                {video_id, question?} -> 202 {job_id, queue_position}
                                                     |  200 {trace_id, cached: true}
  GET  /jobs/{job_id}         -> staged progress
  GET  /traces/{trace_id}     -> trace JSON (schema v1)
  GET  /library              -> {items: [...]}
  GET  /videos/{video_id}/file -> raw video bytes

POST /videos enforces the M5 input limits with friendly JSON errors:
  * over MAX_UPLOAD_BYTES (100 MB)          -> 413
  * probeable duration over MAX_DURATION_S  -> 422
  * undecodable / duration unprobeable      -> 415 (likely unsupported codec)

Importing this module builds the app and its job-queue worker thread but does
NOT load the model or touch CUDA — that happens lazily inside the first job.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from jsr.server.ids import DEFAULT_QUESTION, trace_id_for, video_id_for
from jsr.server.jobs import JobQueue
from jsr.server.pipeline import Pipeline
from jsr.server.store import TraceStore, VideoStore


# Input limits (M5). MVP targets short clips (SPEC: 5-20 s, 480p); these caps
# keep a single trace pass inside the VRAM/token budget and give the user a
# friendly error instead of an OOM deep inside a job.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_DURATION_S = 25.0  # hard clip-length cap


class TraceRequest(BaseModel):
    video_id: str
    question: str | None = None


def default_probe_duration(data: bytes, filename: str) -> float:
    """Container duration in seconds, or 0.0 if it can't be probed cheaply."""
    try:
        import av

        with av.open(io.BytesIO(data)) as container:
            if container.duration:
                return round(float(container.duration / av.time_base), 3)
            stream = container.streams.video[0]
            if stream.duration and stream.time_base:
                return round(float(stream.duration * stream.time_base), 3)
    except Exception:  # noqa: BLE001 - probing must never block an upload
        return 0.0
    return 0.0


def create_app(
    *,
    uploads_dir: str | Path = "uploads",
    traces_dir: str | Path = "traces",
    video_store: VideoStore | None = None,
    trace_store: TraceStore | None = None,
    pipeline: Pipeline | None = None,
    job_queue: JobQueue | None = None,
    probe_fn: Callable[[bytes, str], float] | None = None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    max_duration_s: float = MAX_DURATION_S,
) -> FastAPI:
    video_store = video_store or VideoStore(uploads_dir)
    trace_store = trace_store or TraceStore(traces_dir)
    probe_fn = probe_fn or default_probe_duration
    if job_queue is None:
        pipeline = pipeline or Pipeline(video_store=video_store, trace_store=trace_store)
        job_queue = JobQueue(runner=pipeline.run)

    app = FastAPI(title="J-Space-Replay backend")
    app.state.video_store = video_store
    app.state.trace_store = trace_store
    app.state.job_queue = job_queue
    app.state.pipeline = pipeline

    @app.post("/videos")
    async def upload_video(file: UploadFile = File(...)):  # noqa: B008 - FastAPI DI idiom
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload — no file bytes received.")
        filename = file.filename or "upload.mp4"

        # -- input limits (M5), friendly errors before anything hits disk ----
        if len(data) > max_upload_bytes:
            mb = len(data) / (1024 * 1024)
            limit_mb = max_upload_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Video is {mb:.0f} MB, over the {limit_mb:.0f} MB limit. Trim the "
                    "clip (this tool targets 5-25 s at 480p) or re-encode it smaller."
                ),
            )
        duration_s = probe_fn(data, filename)
        if duration_s <= 0:
            raise HTTPException(
                status_code=415,
                detail=(
                    "Could not decode this video — its duration was unreadable, which "
                    "usually means an unsupported or corrupt codec. Please upload a "
                    "standard H.264/H.265 MP4 or VP9/AV1 WebM clip."
                ),
            )
        if duration_s > max_duration_s:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Clip is {duration_s:.1f} s, over the {max_duration_s:.0f} s limit. "
                    "Please upload a shorter clip — this tool is built for short "
                    "(5-25 s) videos so a trace fits the GPU/token budget."
                ),
            )

        video_id = video_id_for(data)
        record = video_store.save(video_id, filename, data, duration_s)
        return {"video_id": record.video_id, "filename": record.filename, "duration_s": record.duration_s}

    @app.post("/traces")
    async def request_trace(req: TraceRequest):
        if video_store.get(req.video_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown video_id {req.video_id}")
        question = req.question if (req.question and req.question.strip()) else DEFAULT_QUESTION
        trace_id = trace_id_for(req.video_id, question)
        if trace_store.has(trace_id):  # cache hit -> instant
            return {"trace_id": trace_id, "cached": True}
        existing = job_queue.find_active(trace_id)
        if existing is not None:  # same (video, question) already in flight
            return JSONResponse(
                status_code=202,
                content={"job_id": existing.id,
                         "queue_position": job_queue.position_of(existing.id) or 0},
            )
        job = job_queue.submit(video_id=req.video_id, question=question, trace_id=trace_id)
        position = job_queue.position_of(job.id) or 0
        return JSONResponse(status_code=202, content={"job_id": job.id, "queue_position": position})

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job_id {job_id}")
        return job.snapshot(job_queue.position_of(job_id))

    @app.get("/traces/{trace_id}")
    async def get_trace(trace_id: str):
        trace = trace_store.get(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail=f"unknown trace_id {trace_id}")
        return trace

    @app.get("/library")
    async def get_library():
        return {"items": trace_store.library()}

    @app.get("/videos/{video_id}/file")
    async def get_video_file(video_id: str):
        record = video_store.get(video_id)
        if record is None or not record.path.exists():
            raise HTTPException(status_code=404, detail=f"unknown video_id {video_id}")
        return FileResponse(record.path, filename=record.filename)

    return app


app = create_app()
