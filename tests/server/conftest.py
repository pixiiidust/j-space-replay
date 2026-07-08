"""Shared helpers for the M3 backend tests.

Hard rule (matches the M3 brief): nothing here loads the real model or touches
CUDA. Every pipeline is driven by a fake `run_trace` that emits the pipeline's
stage callbacks and returns a real fixture trace from fixtures/traces/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_TRACE = REPO_ROOT / "fixtures" / "traces" / "ball_drop.trace.json"


def load_fixture_trace() -> dict:
    return json.loads(FIXTURE_TRACE.read_text(encoding="utf-8"))


@pytest.fixture
def fixture_trace() -> dict:
    return load_fixture_trace()


def make_fake_run_trace(trace: dict | None = None, *, calls: list | None = None):
    """A stand-in for jsr.trace.run_trace: fires the three real stage callbacks
    and returns a copy of a fixture trace. Records call kwargs if `calls` given."""

    def fake_run_trace(clip, question, *, model, processor, fps, max_pixels, on_stage=None, **kw):
        if calls is not None:
            calls.append({"fps": fps, "max_pixels": max_pixels, "question": question})
        for stage in ("sampling", "prefill_and_generate", "lens_decode"):
            if on_stage:
                on_stage(stage)
        out = json.loads(json.dumps(trace if trace is not None else load_fixture_trace()))
        out["question"] = question
        return out

    return fake_run_trace


@pytest.fixture
def run_trace_factory():
    """Fixture wrapper around make_fake_run_trace (avoids cross-module imports)."""
    return make_fake_run_trace


@pytest.fixture
def sentinel_model():
    """A (model, processor) pair that must never be called for GPU work."""
    return (object(), object())
