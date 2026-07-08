"""Single-GPU job queue and job state.

Strictly one job executes at a time (the GPU can only run one trace pass) — a
background worker thread pulls jobs FIFO. A job submitted while another runs is
"queued" and reports its `queue_position` (0 == running/front). Positions are
computed live from the active list so they shift down as jobs finish.

The queue itself knows nothing about the model; it just serializes a `runner`
callable. The real runner is `pipeline.Pipeline.run`; tests inject a fake.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

# Canonical stage order reported by GET /jobs (API contract, pinned).
STAGES = ["sampling", "prefill_capture", "generating", "lens_decode", "labels", "grounding", "done"]

# run_trace's on_stage names -> the job stage each one begins. Entering a stage
# marks every earlier stage done, so run_trace's single "prefill_and_generate"
# callback covers both prefill_capture and generating.
RUN_TRACE_ENTRY = {
    "sampling": "sampling",
    "prefill_and_generate": "prefill_capture",
    "lens_decode": "lens_decode",
}


@dataclass
class Job:
    id: str
    video_id: str
    question: str
    trace_id: str
    status: str = "queued"  # queued | running | done | error
    stage: str | None = None
    stages_done: list[str] = field(default_factory=list)
    error: str | None = None
    warning: str | None = None
    created_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def enter(self, stage: str) -> None:
        """Begin `stage`, marking every earlier canonical stage done."""
        idx = STAGES.index(stage)
        with self._lock:
            for s in STAGES[:idx]:
                if s not in self.stages_done:
                    self.stages_done.append(s)
            self.stage = stage

    def complete_stage(self, stage: str) -> None:
        with self._lock:
            if stage not in self.stages_done:
                self.stages_done.append(stage)

    def snapshot(self, queue_position: int | None) -> dict:
        with self._lock:
            out: dict = {
                "job_id": self.id,
                "status": self.status,
                "stages_done": list(self.stages_done),
            }
            if queue_position is not None and self.status in ("queued", "running"):
                out["queue_position"] = queue_position
            if self.stage is not None:
                out["stage"] = self.stage
            if self.trace_id and self.status == "done":
                out["trace_id"] = self.trace_id
            if self.error:
                out["error"] = self.error
            if self.warning:
                out["warning"] = self.warning
            return out


class JobQueue:
    def __init__(self, runner: Callable[[Job], None]):
        self._runner = runner
        self._jobs: dict[str, Job] = {}
        self._active: list[str] = []  # not-yet-finished job ids, FIFO; [0] runs
        self._cond = threading.Condition()
        self._stop = False
        self._worker = threading.Thread(target=self._loop, name="jsr-job-worker", daemon=True)
        self._worker.start()

    def submit(self, *, video_id: str, question: str, trace_id: str) -> Job:
        job = Job(id=uuid.uuid4().hex, video_id=video_id, question=question, trace_id=trace_id)
        with self._cond:
            self._jobs[job.id] = job
            self._active.append(job.id)
            self._cond.notify_all()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def find_active(self, trace_id: str) -> Job | None:
        """The queued/running job already computing `trace_id`, if any —
        lets POST /traces dedupe instead of double-running the GPU."""
        with self._cond:
            for job_id in self._active:
                job = self._jobs[job_id]
                if job.trace_id == trace_id:
                    return job
        return None

    def position_of(self, job_id: str) -> int | None:
        """0 == front (running or next); None once the job has finished."""
        with self._cond:
            try:
                return self._active.index(job_id)
            except ValueError:
                return None

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._stop and not self._active:
                    self._cond.wait()
                if self._stop:
                    return
                job = self._jobs[self._active[0]]
                job.status = "running"
            try:
                self._runner(job)
                with self._cond:
                    if job.status != "error":
                        job.status = "done"
            except Exception as exc:  # noqa: BLE001 - surface any failure as job error
                with self._cond:
                    job.status = "error"
                    if not job.error:
                        job.error = str(exc)
            finally:
                with self._cond:
                    if self._active and self._active[0] == job.id:
                        self._active.pop(0)
                    self._cond.notify_all()

    def shutdown(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
