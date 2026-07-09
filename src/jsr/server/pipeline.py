"""The real job runner: run_trace -> labels -> grounding -> store, with the
Milestone-3 failure paths (OOM auto-retry, corrupt-video clear error).

Every heavy dependency (torch, the model, run_trace, jsr.labels) is imported
lazily so importing this module — and constructing a `Pipeline` — never loads
the model or touches CUDA. The model is loaded once, on the first job.

All the moving parts are injectable so tests drive the whole runner with fakes:
`run_trace_fn`, `load_model`, `grounding_query_factory`, `oom_errors`,
`video_errors`, and `add_concepts` can each be replaced.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from jsr.server.grounding import make_query_fn, run_grounding
from jsr.server.jobs import RUN_TRACE_ENTRY, Job
from jsr.server.store import TraceStore, VideoStore

# Reduced settings used for the single OOM auto-retry (SPEC / PLAN M3).
RETRY_FPS = 0.5
DEFAULT_MAX_PIXELS = 250 * 28 * 28  # mirror jsr.model.MAX_PIXELS without importing torch


def _default_oom_errors() -> tuple[type[BaseException], ...]:
    try:
        import torch

        return (torch.cuda.OutOfMemoryError,)
    except Exception:  # noqa: BLE001 - torch/CUDA unavailable: nothing to catch
        return ()


def _default_video_errors() -> tuple[type[BaseException], ...]:
    errs: list[type[BaseException]] = [ValueError, OSError]
    try:
        import av

        errs.append(av.error.FFmpegError)
    except Exception:  # noqa: BLE001
        pass
    return tuple(errs)


class CorruptVideoError(Exception):
    """Raised when the uploaded video cannot be decoded — surfaced verbatim to the client."""


class Pipeline:
    def __init__(
        self,
        *,
        video_store: VideoStore,
        trace_store: TraceStore,
        run_trace_fn: Callable | None = None,
        load_model: Callable | None = None,
        grounding_query_factory: Callable = make_query_fn,
        oom_errors: tuple[type[BaseException], ...] | None = None,
        video_errors: tuple[type[BaseException], ...] | None = None,
        add_concepts: Callable | None = None,
        preflight_fn: Callable[[], None] | None = None,
        default_fps: float = 1.0,
        default_max_pixels: int = DEFAULT_MAX_PIXELS,
    ):
        self.video_store = video_store
        self.trace_store = trace_store
        self._run_trace_fn = run_trace_fn
        self._load_model = load_model
        self._grounding_query_factory = grounding_query_factory
        self._oom_errors = _default_oom_errors() if oom_errors is None else oom_errors
        self._video_errors = _default_video_errors() if video_errors is None else video_errors
        self._add_concepts = add_concepts  # None -> import jsr.labels lazily per run
        self._preflight_fn = preflight_fn  # None -> soft VRAM guard (preflight.preflight_job)
        self.default_fps = default_fps
        self.default_max_pixels = default_max_pixels
        self._model = None  # lazily loaded (model, processor) tuple, cached
        self._jlens = None  # lazily loaded JLens, cached (686 MB fp16 on GPU)

    # -- VRAM pre-flight (soft: refuses only when measurably below budget) --
    def _preflight(self) -> None:
        if self._preflight_fn is not None:
            self._preflight_fn()
            return
        from jsr.server.preflight import preflight_job

        preflight_job()

    # -- lazy model load ---------------------------------------------------
    def _get_model(self):
        if self._model is None:
            self._preflight()
            loader = self._load_model
            if loader is None:
                from jsr.model import load_model_and_processor

                loader = load_model_and_processor
            self._model = loader()
        return self._model

    def warmup(self) -> None:
        self._get_model()

    def _get_jlens(self):
        """The fitted J-lens, loaded once. FileNotFoundError (with the
        regenerate-command guidance from JLens.load) surfaces as the job's
        error message — the UI only offers the lens when /lenses says it
        exists, so hitting this means the file vanished mid-session."""
        if self._jlens is None:
            from jsr.lens import JLens

            self._jlens = JLens.load()
        return self._jlens

    def _run_trace(self):
        if self._run_trace_fn is not None:
            return self._run_trace_fn
        from jsr.trace import run_trace

        return run_trace

    def _resolve_add_concepts(self) -> Callable | None:
        if self._add_concepts is not None:
            return self._add_concepts
        try:  # M2, being written in parallel — skip cleanly until it lands.
            from jsr.labels import add_concepts

            return add_concepts
        except ImportError:
            return None

    # -- the runner --------------------------------------------------------
    def run(self, job: Job) -> None:
        model, processor = self._get_model()
        record = self.video_store.get(job.video_id)
        if record is None:
            raise CorruptVideoError(f"video {job.video_id} not found on disk")
        clip = record.path

        trace = self._trace_with_failure_paths(job, clip, model, processor)

        # -- labels stage (M2; skipped if jsr.labels not importable) --------
        job.enter("labels")
        add_concepts = self._resolve_add_concepts()
        if add_concepts is not None:
            # clip enables jsr.labels' caption pass (extra candidate phrases)
            add_concepts(trace, model=model, processor=processor, clip=clip)
        job.complete_stage("labels")

        # -- grounding stage (model-native boxes for top ~5 concepts) -------
        job.enter("grounding")
        query_fn = self._grounding_query_factory(model, processor, clip)
        run_grounding(trace, query_fn=query_fn, top_k=5)
        job.complete_stage("grounding")

        # -- persist (atomic) then finish -----------------------------------
        job.enter("done")
        lib_item = {
            "trace_id": job.trace_id,
            "video_id": job.video_id,
            "question": job.question,
            "lens": trace.get("meta", {}).get("lens", "logit-lens-v1"),
            "answer": trace.get("answer", ""),
            "created_at": _now_iso(),
            "duration_s": record.duration_s,
        }
        self.trace_store.put(job.trace_id, trace, lib_item)
        job.complete_stage("done")
        job.status = "done"

    def _trace_with_failure_paths(self, job: Job, clip: Path, model, processor) -> dict:
        run_trace = self._run_trace()
        lens_kw = {}
        if getattr(job, "lens", "logit-lens-v1") == "j-lens-v1":
            lens_kw["jlens"] = self._get_jlens()

        def on_stage(name: str) -> None:
            entry = RUN_TRACE_ENTRY.get(name)
            if entry is not None:
                job.enter(entry)

        try:
            return run_trace(
                clip,
                job.question,
                model=model,
                processor=processor,
                fps=self.default_fps,
                max_pixels=self.default_max_pixels,
                on_stage=on_stage,
                **lens_kw,
            )
        except self._oom_errors:
            # OOM path: retry ONCE at half the pixels and lower FPS, with a warning.
            job.warning = (
                "Ran out of GPU memory; automatically retried once at reduced "
                f"resolution (max_pixels {self.default_max_pixels // 2}) and "
                f"{RETRY_FPS} FPS. Results may be coarser."
            )
            return run_trace(
                clip,
                job.question,
                model=model,
                processor=processor,
                fps=RETRY_FPS,
                max_pixels=self.default_max_pixels // 2,
                on_stage=on_stage,
                **lens_kw,
            )
            # A second OOM here is not retried again; it propagates to the queue
            # and surfaces as a plain job error.
        except self._video_errors as exc:
            raise CorruptVideoError(
                "Could not decode the uploaded video — it may be corrupt or in an "
                f"unsupported codec. ({exc})"
            ) from exc


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
