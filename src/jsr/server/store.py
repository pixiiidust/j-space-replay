"""On-disk stores for uploaded videos and computed traces.

Crash safety (Milestone 3 exit criterion "kill -9 during a job leaves no
corrupt state"): every write goes to a `<name>.tmp` file first and is then
promoted with `os.replace`, which is atomic on Windows and POSIX. A process
killed mid-write leaves only a `.tmp` file — never a half-written trace under
its real name. `_clean_temp_files` runs at construction and deletes any stray
`.tmp` left by a previous crash, so readers only ever see complete files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TMP_SUFFIX = ".tmp"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + TMP_SUFFIX)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic promotion


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def _clean_temp_files(directory: Path) -> list[Path]:
    removed = []
    for p in directory.glob("*" + TMP_SUFFIX):
        p.unlink()
        removed.append(p)
    return removed


@dataclass
class VideoRecord:
    video_id: str
    filename: str
    duration_s: float
    path: Path


class VideoStore:
    """Uploaded video files, keyed by content hash (video_id).

    Stored as `<video_id><ext>` plus a tiny `<video_id>.meta.json` sidecar so
    `/videos/{id}/file` and `/library` never have to re-probe the container.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        _clean_temp_files(self.root)

    def _meta_path(self, video_id: str) -> Path:
        return self.root / f"{video_id}.meta.json"

    def save(self, video_id: str, filename: str, data: bytes, duration_s: float) -> VideoRecord:
        ext = Path(filename).suffix or ".mp4"
        path = self.root / f"{video_id}{ext}"
        _atomic_write_bytes(path, data)
        meta = {"video_id": video_id, "filename": filename, "duration_s": duration_s, "ext": ext}
        _atomic_write_json(self._meta_path(video_id), meta)
        return VideoRecord(video_id, filename, duration_s, path)

    def get(self, video_id: str) -> VideoRecord | None:
        mp = self._meta_path(video_id)
        if not mp.exists():
            return None
        meta = json.loads(mp.read_text(encoding="utf-8"))
        path = self.root / f"{video_id}{meta['ext']}"
        if not path.exists():
            return None
        return VideoRecord(video_id, meta["filename"], meta["duration_s"], path)


class TraceStore:
    """Computed traces keyed by trace_id = f"{video_id}-{q_hash}".

    The full (large) trace lives in `<trace_id>.json`; a small
    `<trace_id>.lib.json` sidecar holds just the library-card fields so
    `/library` stays cheap. Cache hit == the `.json` exists.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # Startup cleanup: drop partial writes left by an interrupted job.
        self.removed_on_start = _clean_temp_files(self.root)

    def _trace_path(self, trace_id: str) -> Path:
        return self.root / f"{trace_id}.json"

    def _lib_path(self, trace_id: str) -> Path:
        return self.root / f"{trace_id}.lib.json"

    def has(self, trace_id: str) -> bool:
        return self._trace_path(trace_id).exists()

    def put(self, trace_id: str, trace: dict, lib_item: dict) -> None:
        # Write the big trace first, then the tiny library sidecar. If we crash
        # between the two, startup cleanup removes any .tmp and the trace simply
        # won't appear in /library until recomputed — never corrupt.
        _atomic_write_json(self._trace_path(trace_id), trace)
        _atomic_write_json(self._lib_path(trace_id), lib_item)

    def get(self, trace_id: str) -> dict | None:
        p = self._trace_path(trace_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def library(self) -> list[dict]:
        items = []
        for p in sorted(self.root.glob("*.lib.json")):
            try:
                items.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        items.sort(key=lambda it: it.get("created_at", ""), reverse=True)
        return items
