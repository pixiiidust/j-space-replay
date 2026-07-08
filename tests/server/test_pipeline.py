"""Pipeline runner: stage mapping, labels/grounding stages, and the M3 failure
paths (OOM auto-retry, corrupt-video clear error) — all with fakes, no model.
"""

from __future__ import annotations

import pytest

from jsr.server.jobs import STAGES, Job
from jsr.server.pipeline import CorruptVideoError, Pipeline
from jsr.server.store import TraceStore, VideoStore


class StubOOM(Exception):
    """Stand-in for torch.cuda.OutOfMemoryError so tests never import CUDA machinery."""


def _stores(tmp_path):
    videos = VideoStore(tmp_path / "uploads")
    traces = TraceStore(tmp_path / "traces")
    videos.save("vid0", "clip.mp4", b"\x00\x01", 12.0)
    return videos, traces


def _job(video_id="vid0", question="What happens?", trace_id="vid0-abc"):
    return Job(id="j1", video_id=video_id, question=question, trace_id=trace_id)


def _noop_labels(trace, **kw):
    """Hermetic stand-in for jsr.labels.add_concepts (which now exists post-M2
    and does real caption/model work when given model+processor+clip)."""
    return trace


def test_full_run_progresses_through_all_stages(tmp_path, sentinel_model, run_trace_factory):
    videos, traces = _stores(tmp_path)
    pipe = Pipeline(
        video_store=videos,
        trace_store=traces,
        run_trace_fn=run_trace_factory(),
        load_model=lambda: sentinel_model,
        add_concepts=_noop_labels,
        grounding_query_factory=lambda m, p, clip: (lambda label, t: "[0,0,1,1]"),
    )
    job = _job()
    pipe.run(job)

    assert job.status == "done"
    assert job.stages_done == STAGES  # every stage, in order
    assert job.stage == "done"
    assert traces.has("vid0-abc")
    # library item is populated from the run
    item = traces.library()[0]
    assert item["trace_id"] == "vid0-abc"
    assert item["video_id"] == "vid0"
    assert item["duration_s"] == 12.0
    assert item["question"] == "What happens?"


def test_labels_stage_skipped_cleanly_when_module_missing(
    tmp_path, sentinel_model, run_trace_factory, monkeypatch
):
    """If jsr.labels can't be imported, the labels stage still runs, no crash.
    (jsr.labels exists since M2 merged, so the absence is simulated.)"""
    videos, traces = _stores(tmp_path)
    pipe = Pipeline(
        video_store=videos,
        trace_store=traces,
        run_trace_fn=run_trace_factory(),
        load_model=lambda: sentinel_model,
        grounding_query_factory=lambda m, p, clip: (lambda label, t: ""),
        add_concepts=None,  # resolve via import ...
    )
    monkeypatch.setattr(Pipeline, "_resolve_add_concepts", lambda self: None)  # ... which "fails"
    job = _job()
    pipe.run(job)
    assert "labels" in job.stages_done
    assert job.status == "done"


def test_labels_then_grounding_populate_the_trace(tmp_path, sentinel_model, run_trace_factory):
    videos, traces = _stores(tmp_path)

    def fake_add_concepts(trace, *, model, processor, clip=None):
        trace["frame_groups"][0]["concepts"] = [{"label": "red ball", "strength": 3.0}]

    pipe = Pipeline(
        video_store=videos,
        trace_store=traces,
        run_trace_fn=run_trace_factory(),
        load_model=lambda: sentinel_model,
        add_concepts=fake_add_concepts,
        grounding_query_factory=lambda m, p, clip: (
            lambda label, t: '[{"bbox_2d": [1, 2, 3, 4], "label": "x"}]'
        ),
    )
    job = _job()
    pipe.run(job)
    stored = traces.get("vid0-abc")
    assert stored["frame_groups"][0]["concepts"][0]["label"] == "red ball"
    assert stored["grounding"][0]["box"] == [1.0, 2.0, 3.0, 4.0]
    assert stored["grounding"][0]["source"] == "model-grounding-query"


def test_oom_retries_once_at_lower_settings_with_warning(
    tmp_path, sentinel_model, run_trace_factory
):
    videos, traces = _stores(tmp_path)
    calls = []
    inner = run_trace_factory(calls=calls)

    state = {"first": True}

    def flaky_run_trace(*args, **kwargs):
        if state["first"]:
            state["first"] = False
            raise StubOOM("CUDA out of memory")
        return inner(*args, **kwargs)

    pipe = Pipeline(
        video_store=videos,
        trace_store=traces,
        run_trace_fn=flaky_run_trace,
        load_model=lambda: sentinel_model,
        oom_errors=(StubOOM,),
        add_concepts=_noop_labels,
        grounding_query_factory=lambda m, p, clip: (lambda label, t: ""),
        default_max_pixels=1000,
        default_fps=1.0,
    )
    job = _job()
    pipe.run(job)

    assert job.status == "done"
    assert job.warning is not None and "memory" in job.warning.lower()
    # Retry used half the pixels and the reduced FPS.
    assert calls[-1]["max_pixels"] == 500
    assert calls[-1]["fps"] == 0.5


def test_corrupt_video_gives_clear_error(tmp_path, sentinel_model):
    videos, traces = _stores(tmp_path)

    def bad_run_trace(*args, **kwargs):
        raise ValueError("no frames decoded from clip.mp4")

    pipe = Pipeline(
        video_store=videos,
        trace_store=traces,
        run_trace_fn=bad_run_trace,
        load_model=lambda: sentinel_model,
        oom_errors=(StubOOM,),
        video_errors=(ValueError,),
    )
    with pytest.raises(CorruptVideoError) as ei:
        pipe.run(_job())
    assert "corrupt" in str(ei.value).lower() or "unsupported" in str(ei.value).lower()


def test_model_loaded_lazily_only_once(tmp_path, sentinel_model, run_trace_factory):
    videos, traces = _stores(tmp_path)
    loads = {"n": 0}

    def loader():
        loads["n"] += 1
        return sentinel_model

    pipe = Pipeline(
        video_store=videos,
        trace_store=traces,
        run_trace_fn=run_trace_factory(),
        load_model=loader,
        add_concepts=_noop_labels,
        grounding_query_factory=lambda m, p, clip: (lambda label, t: ""),
    )
    assert loads["n"] == 0  # not loaded at construction
    pipe.run(_job(trace_id="vid0-a"))
    pipe.run(_job(trace_id="vid0-b"))
    assert loads["n"] == 1  # loaded once, cached across jobs
