"""FastAPI endpoint contract tests via TestClient.

Every app is wired with a fake pipeline (fake run_trace, fake grounding) and a
fake duration probe, so nothing loads the model or touches CUDA. Covers the M3
exit criteria at the HTTP layer: concurrent queueing, and instant cache re-hit.
"""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from jsr.server.api import create_app
from jsr.server.jobs import JobQueue
from jsr.server.pipeline import Pipeline
from jsr.server.store import TraceStore, VideoStore


def _fake_pipeline(tmp_path, run_trace_fn, sentinel=(object(), object())):
    videos = VideoStore(tmp_path / "uploads")
    traces = TraceStore(tmp_path / "traces")
    pipe = Pipeline(
        video_store=videos,
        trace_store=traces,
        run_trace_fn=run_trace_fn,
        load_model=lambda: sentinel,
        add_concepts=lambda trace, **kw: trace,  # hermetic: real jsr.labels does model work
        grounding_query_factory=lambda m, p, clip: (lambda label, t: ""),
    )
    return videos, traces, pipe


def _client(tmp_path, run_trace_fn):
    videos, traces, pipe = _fake_pipeline(tmp_path, run_trace_fn)
    app = create_app(
        video_store=videos,
        trace_store=traces,
        pipeline=pipe,
        probe_fn=lambda data, filename: 12.5,
    )
    return TestClient(app)


def _poll_job(client, job_id, want="done", timeout=5.0):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        last = client.get(f"/jobs/{job_id}").json()
        if last["status"] in (want, "error"):
            return last
        time.sleep(0.02)
    return last


def test_upload_returns_contract_shape(tmp_path, run_trace_factory):
    with _client(tmp_path, run_trace_factory()) as client:
        r = client.post("/videos", files={"file": ("clip.mp4", b"\x00\x01\x02\x03", "video/mp4")})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"video_id", "filename", "duration_s"}
        assert len(body["video_id"]) == 16  # sha256 first 16 hex
        assert body["filename"] == "clip.mp4"
        assert body["duration_s"] == 12.5


def test_full_trace_flow_then_cache_hit(tmp_path, run_trace_factory):
    with _client(tmp_path, run_trace_factory()) as client:
        vid = client.post(
            "/videos", files={"file": ("clip.mp4", b"hello-bytes", "video/mp4")}
        ).json()["video_id"]

        r = client.post("/traces", json={"video_id": vid, "question": "What happens?"})
        assert r.status_code == 202
        job = r.json()
        assert "job_id" in job and job["queue_position"] == 0

        final = _poll_job(client, job["job_id"])
        assert final["status"] == "done", final
        assert final["stages_done"] == [
            "sampling", "prefill_capture", "generating", "lens_decode", "labels", "grounding", "done",
        ]
        trace_id = final["trace_id"]

        # trace fetchable and schema-valid
        tr = client.get(f"/traces/{trace_id}")
        assert tr.status_code == 200
        assert tr.json()["question"] == "What happens?"

        # library lists it
        lib = client.get("/library").json()["items"]
        assert any(it["trace_id"] == trace_id for it in lib)

        # re-request the SAME (video, question) -> instant cache hit, no new job
        r2 = client.post("/traces", json={"video_id": vid, "question": "What happens?"})
        assert r2.status_code == 200
        assert r2.json() == {"trace_id": trace_id, "cached": True}


def test_default_question_used_when_omitted(tmp_path, run_trace_factory):
    with _client(tmp_path, run_trace_factory()) as client:
        vid = client.post(
            "/videos", files={"file": ("c.mp4", b"abc", "video/mp4")}
        ).json()["video_id"]
        r = client.post("/traces", json={"video_id": vid})
        assert r.status_code == 202
        final = _poll_job(client, r.json()["job_id"])
        assert final["status"] == "done"
        trace = client.get(f"/traces/{final['trace_id']}").json()
        assert trace["question"] == "Describe what happens in this video."


def test_unknown_video_and_trace_and_job_404(tmp_path, run_trace_factory):
    with _client(tmp_path, run_trace_factory()) as client:
        assert client.post("/traces", json={"video_id": "nope"}).status_code == 404
        assert client.get("/traces/nope").status_code == 404
        assert client.get("/jobs/nope").status_code == 404
        assert client.get("/videos/nope/file").status_code == 404


def test_video_file_download_roundtrip(tmp_path, run_trace_factory):
    with _client(tmp_path, run_trace_factory()) as client:
        payload = b"\x00\x11\x22raw-video-bytes"
        vid = client.post(
            "/videos", files={"file": ("v.mp4", payload, "video/mp4")}
        ).json()["video_id"]
        r = client.get(f"/videos/{vid}/file")
        assert r.status_code == 200
        assert r.content == payload


def test_two_concurrent_trace_requests_queue_correctly(tmp_path):
    """M3 exit criterion at the HTTP layer: second request queues behind the first."""
    videos = VideoStore(tmp_path / "uploads")
    traces = TraceStore(tmp_path / "traces")
    started = threading.Event()
    release = threading.Event()

    def slow_runner(job):
        started.set()
        release.wait(timeout=5)
        job.status = "done"

    queue = JobQueue(runner=slow_runner)
    app = create_app(
        video_store=videos,
        trace_store=traces,
        job_queue=queue,
        probe_fn=lambda data, filename: 10.0,
    )
    try:
        with TestClient(app) as client:
            vid = client.post(
                "/videos", files={"file": ("c.mp4", b"xyz", "video/mp4")}
            ).json()["video_id"]

            r1 = client.post("/traces", json={"video_id": vid, "question": "q1"})
            assert r1.status_code == 202
            assert started.wait(timeout=5), "first job never started"
            j1 = r1.json()["job_id"]
            assert client.get(f"/jobs/{j1}").json()["status"] == "running"

            r2 = client.post("/traces", json={"video_id": vid, "question": "q2"})
            assert r2.status_code == 202
            assert r2.json()["queue_position"] >= 1
            snap2 = client.get(f"/jobs/{r2.json()['job_id']}").json()
            assert snap2["status"] == "queued"
            assert snap2["queue_position"] >= 1

            release.set()
            final2 = _poll_job(client, r2.json()["job_id"])
            assert final2["status"] == "done"
    finally:
        release.set()
        queue.shutdown()


def test_same_video_question_resubmit_returns_existing_job(tmp_path, run_trace_factory):
    """POST /traces twice for the SAME (video, question) while the first job is
    still in flight must return the existing job, not double-run the GPU."""
    gate = threading.Event()

    def slow_run_trace(clip, question, *, on_stage=None, **kw):
        gate.wait(timeout=5.0)
        return run_trace_factory()(clip, question, on_stage=on_stage, **kw)

    with _client(tmp_path, slow_run_trace) as client:
        vid = client.post(
            "/videos", files={"file": ("clip.mp4", b"dedup-bytes", "video/mp4")}
        ).json()["video_id"]
        first = client.post("/traces", json={"video_id": vid, "question": "Q?"})
        second = client.post("/traces", json={"video_id": vid, "question": "Q?"})
        assert first.status_code == 202 and second.status_code == 202
        assert second.json()["job_id"] == first.json()["job_id"]
        gate.set()
        final = _poll_job(client, first.json()["job_id"])
        assert final["status"] == "done"


def test_lens_field_routes_to_jlens_and_gets_distinct_trace_id(tmp_path, run_trace_factory):
    """lens=j-lens-v1 must reach run_trace as a jlens object, produce a
    distinct -jl trace_id (never clobbering the logit trace of the same
    question), and stamp the lens on the library card."""
    seen_jlens: list = []

    def recording_run_trace(clip, question, *, on_stage=None, **kw):
        seen_jlens.append(kw.get("jlens"))
        out = run_trace_factory()(clip, question, on_stage=on_stage, **kw)
        if kw.get("jlens") is not None:
            out["meta"]["lens"] = "j-lens-v1"
        return out

    videos, traces, pipe = _fake_pipeline(tmp_path, recording_run_trace)
    pipe._jlens = object()  # hermetic: no fitted lens file in the test env
    app = create_app(video_store=videos, trace_store=traces, pipeline=pipe,
                     probe_fn=lambda data, filename: 12.5)
    with TestClient(app) as client:
        vid = client.post(
            "/videos", files={"file": ("clip.mp4", b"lens-bytes", "video/mp4")}
        ).json()["video_id"]

        r_logit = client.post("/traces", json={"video_id": vid, "question": "Q?"})
        final_logit = _poll_job(client, r_logit.json()["job_id"])
        r_jl = client.post("/traces", json={"video_id": vid, "question": "Q?", "lens": "j-lens-v1"})
        assert r_jl.status_code == 202, "j-lens request must be a NEW job, not the logit cache"
        final_jl = _poll_job(client, r_jl.json()["job_id"])

        assert final_logit["status"] == final_jl["status"] == "done"
        assert final_jl["trace_id"] == final_logit["trace_id"] + "-jl"
        assert seen_jlens[0] is None and seen_jlens[1] is not None

        lib = {it["trace_id"]: it for it in client.get("/library").json()["items"]}
        assert lib[final_jl["trace_id"]]["lens"] == "j-lens-v1"
        assert lib[final_logit["trace_id"]]["lens"] == "logit-lens-v1"

        # each lens hits its own cache
        again = client.post("/traces", json={"video_id": vid, "question": "Q?", "lens": "j-lens-v1"})
        assert again.status_code == 200 and again.json()["cached"] is True


def test_unknown_lens_rejected_422(tmp_path, run_trace_factory):
    with _client(tmp_path, run_trace_factory()) as client:
        vid = client.post(
            "/videos", files={"file": ("c.mp4", b"zz", "video/mp4")}
        ).json()["video_id"]
        r = client.post("/traces", json={"video_id": vid, "lens": "tuned-lens-v9"})
        assert r.status_code == 422
        assert "tuned-lens-v9" in r.json()["detail"]


def test_lenses_endpoint_lists_default(tmp_path, run_trace_factory):
    with _client(tmp_path, run_trace_factory()) as client:
        body = client.get("/lenses").json()
        assert body["default"] == "logit-lens-v1"
        assert "logit-lens-v1" in body["lenses"]  # j-lens presence depends on a fitted file
