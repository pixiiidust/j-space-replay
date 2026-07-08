"""Job queue: strictly one GPU job at a time; queued jobs report position.

Proves the M3 exit criterion "two traces requested concurrently queue
correctly" at the queue level, with a slow fake runner (no model).
"""

from __future__ import annotations

import threading
import time

import pytest

from jsr.server.jobs import STAGES, Job, JobQueue


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_second_job_queues_behind_a_running_one():
    release = threading.Event()
    started = threading.Event()

    def slow_runner(job: Job):
        started.set()
        assert release.wait(timeout=5), "runner never released"

    q = JobQueue(runner=slow_runner)
    try:
        j1 = q.submit(video_id="v", question="q1", trace_id="v-q1")
        assert _wait(started.is_set), "first job never started"
        assert q.get(j1.id).status == "running"
        assert q.position_of(j1.id) == 0

        j2 = q.submit(video_id="v", question="q2", trace_id="v-q2")
        # Second job is strictly behind the running one.
        assert q.position_of(j2.id) >= 1
        assert q.get(j2.id).status == "queued"
        assert j2.snapshot(q.position_of(j2.id))["queue_position"] >= 1

        release.set()
        assert _wait(lambda: q.get(j2.id).status == "done"), "second job never ran"
        assert q.get(j1.id).status == "done"
        # Positions clear out once finished.
        assert q.position_of(j1.id) is None
        assert q.position_of(j2.id) is None
    finally:
        release.set()
        q.shutdown()


def test_only_one_job_runs_at_a_time():
    concurrent = 0
    max_seen = 0
    lock = threading.Lock()
    go = threading.Event()

    def runner(job: Job):
        nonlocal concurrent, max_seen
        with lock:
            concurrent += 1
            max_seen = max(max_seen, concurrent)
        go.wait(timeout=2)
        with lock:
            concurrent -= 1

    q = JobQueue(runner=runner)
    try:
        ids = [q.submit(video_id="v", question=f"q{i}", trace_id=f"v-q{i}").id for i in range(4)]
        time.sleep(0.1)
        go.set()
        assert _wait(lambda: all(q.get(i).status == "done" for i in ids))
        assert max_seen == 1  # never more than one job executing
    finally:
        go.set()
        q.shutdown()


def test_runner_exception_becomes_job_error():
    def boom(job: Job):
        raise RuntimeError("kaboom")

    q = JobQueue(runner=boom)
    try:
        j = q.submit(video_id="v", question="q", trace_id="v-q")
        assert _wait(lambda: q.get(j.id).status == "error")
        assert "kaboom" in q.get(j.id).error
    finally:
        q.shutdown()


def test_job_enter_marks_prior_stages_done():
    job = Job(id="x", video_id="v", question="q", trace_id="v-q")
    job.enter("lens_decode")
    # Entering lens_decode implies sampling, prefill_capture, generating are done.
    assert job.stages_done == ["sampling", "prefill_capture", "generating"]
    assert job.stage == "lens_decode"
    assert STAGES.index("lens_decode") == 3


@pytest.mark.parametrize("q_pos", [None, 3])
def test_snapshot_shape(q_pos):
    job = Job(id="x", video_id="v", question="q", trace_id="v-q")
    job.status = "queued"
    snap = job.snapshot(q_pos)
    assert snap["job_id"] == "x"
    assert snap["status"] == "queued"
    assert snap["stages_done"] == []
    if q_pos is not None:
        assert snap["queue_position"] == q_pos
