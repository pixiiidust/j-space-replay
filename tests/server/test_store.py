"""Trace/video store: atomic writes and crash safety.

Proves the M3 exit criterion "killing the process mid-job leaves no corrupt
state": a job interrupted mid-write leaves only a `.tmp` file, and startup
cleanup removes it so readers never see a partial trace.
"""

from __future__ import annotations

import json

from jsr.server.store import TMP_SUFFIX, TraceStore, VideoStore, _atomic_write_json


def _lib_item(trace_id="v-q"):
    return {
        "trace_id": trace_id,
        "video_id": "v",
        "question": "q",
        "answer": "a",
        "created_at": "2026-07-08T00:00:00Z",
        "duration_s": 12.0,
    }


def test_put_and_get_roundtrip(tmp_path):
    store = TraceStore(tmp_path)
    trace = {"schema": 1, "answer": "hi"}
    store.put("v-q", trace, _lib_item())
    assert store.has("v-q")
    assert store.get("v-q") == trace
    assert store.get("missing") is None


def test_library_lists_only_committed_traces(tmp_path):
    store = TraceStore(tmp_path)
    store.put("v-a", {"schema": 1}, _lib_item("v-a"))
    store.put("v-b", {"schema": 1}, _lib_item("v-b"))
    items = store.library()
    assert {it["trace_id"] for it in items} == {"v-a", "v-b"}


def test_interrupted_write_leaves_only_a_tmp_file(tmp_path):
    """Simulate a crash mid-write: a stray `.tmp` exists, but no real trace."""
    store = TraceStore(tmp_path)
    trace_path = tmp_path / "v-q.json"
    tmp_leftover = tmp_path / ("v-q.json" + TMP_SUFFIX)
    tmp_leftover.write_text('{"partial": ', encoding="utf-8")  # truncated, corrupt

    assert not trace_path.exists()  # os.replace never happened
    assert not store.has("v-q")  # cache miss, not a corrupt hit
    assert store.get("v-q") is None
    assert store.library() == []  # partial trace is invisible to the library


def test_startup_cleanup_removes_stray_tmp_files(tmp_path):
    tmp_leftover = tmp_path / ("v-q.json" + TMP_SUFFIX)
    tmp_leftover.write_text('{"partial": ', encoding="utf-8")
    assert tmp_leftover.exists()

    store = TraceStore(tmp_path)  # cleanup runs in __init__

    assert not tmp_leftover.exists()
    assert tmp_leftover.name.replace(TMP_SUFFIX, "") not in [p.name for p in tmp_path.iterdir()]
    assert store.removed_on_start  # reported what it swept


def test_atomic_write_never_exposes_partial_content(tmp_path):
    """After _atomic_write_json the file is either absent or fully valid JSON."""
    path = tmp_path / "x.json"
    _atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2, 3]}
    assert not (tmp_path / ("x.json" + TMP_SUFFIX)).exists()


def test_video_store_save_and_get(tmp_path):
    store = VideoStore(tmp_path)
    rec = store.save("abcd1234", "clip.mp4", b"\x00\x01\x02", 15.0)
    assert rec.path.exists()
    got = store.get("abcd1234")
    assert got is not None
    assert got.filename == "clip.mp4"
    assert got.duration_s == 15.0
    assert store.get("nope") is None
