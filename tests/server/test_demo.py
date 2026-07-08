"""Demo seeding (M5): the bundled fixtures land in the stores as an instantly
browsable library, with no GPU and no real clip render (make_fixtures faked)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from jsr.server.api import create_app
from jsr.server.demo import repo_root, seed_demo
from jsr.server.ids import trace_id_for
from jsr.server.store import TraceStore, VideoStore

REAL_TRACES = repo_root() / "fixtures" / "traces"


def _staged_root(tmp_path: Path) -> Path:
    """A temp repo root holding copies of the real fixture traces."""
    dst = tmp_path / "fixtures" / "traces"
    dst.mkdir(parents=True)
    for p in REAL_TRACES.glob("*.trace.json"):
        shutil.copy(p, dst / p.name)
    return tmp_path


def _fake_make_fixtures(clips_dir: Path) -> None:
    # Stand in for scripts/make_fixtures.py: write tiny placeholder clip files
    # so ensure_clips is satisfied without rendering video.
    for name in ("ball_drop", "shape_morph", "traffic"):
        (clips_dir / f"{name}.mp4").write_bytes(b"\x00fake-mp4")


def test_seed_demo_populates_both_stores(tmp_path):
    root = _staged_root(tmp_path)
    videos = VideoStore(tmp_path / "uploads")
    traces = TraceStore(tmp_path / "traces")

    seeded = seed_demo(videos, traces, root=root, make_fixtures=_fake_make_fixtures)

    assert len(seeded) == 3
    for name in ("ball_drop", "shape_morph", "traffic"):
        trace = json.loads((root / "fixtures" / "traces" / f"{name}.trace.json").read_text("utf-8"))
        tid = trace_id_for(trace["video_id"], trace["question"])
        assert tid in seeded
        assert traces.has(tid)
        assert videos.get(trace["video_id"]) is not None
    # library lists all three, each with a non-empty answer
    lib = traces.library()
    assert len(lib) == 3
    assert all(item["answer"] for item in lib)


def test_seed_demo_is_idempotent(tmp_path):
    root = _staged_root(tmp_path)
    videos = VideoStore(tmp_path / "uploads")
    traces = TraceStore(tmp_path / "traces")
    seed_demo(videos, traces, root=root, make_fixtures=_fake_make_fixtures)
    seed_demo(videos, traces, root=root, make_fixtures=_fake_make_fixtures)  # again
    assert len(traces.library()) == 3


def test_demo_library_served_over_http(tmp_path):
    root = _staged_root(tmp_path)
    app = create_app(uploads_dir=tmp_path / "uploads", traces_dir=tmp_path / "traces")
    seed_demo(app.state.video_store, app.state.trace_store,
              root=root, make_fixtures=_fake_make_fixtures)
    with TestClient(app) as client:
        lib = client.get("/library").json()["items"]
        assert len(lib) == 3
        tid = lib[0]["trace_id"]
        tr = client.get(f"/traces/{tid}")
        assert tr.status_code == 200
        assert tr.json()["schema"] == 1
