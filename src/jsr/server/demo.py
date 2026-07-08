"""Demo seeding for `jsr up --demo` (Milestone 5).

Pre-seeds the video + trace stores from the committed fixtures so the app
demos instantly with zero GPU work:

  * fixtures/traces/*.trace.json — real pre-baked traces (schema v1, WITH the
    M2 experimental concepts) computed on a GPU and committed to the repo.
  * fixtures/clips/*.mp4 — the matching demo clips. They are tiny (~50 KB) and
    deterministic, so if they are missing we regenerate them from
    scripts/make_fixtures.py at startup rather than committing binaries.

Everything is injectable (repo root, the fixtures loader, the make-fixtures
callable) so it is unit-testable without a checkout layout or a real render.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from jsr.server.ids import trace_id_for
from jsr.server.store import TraceStore, VideoStore


def repo_root() -> Path:
    """Repo root, assuming the usual `src/jsr/server/demo.py` checkout layout."""
    return Path(__file__).resolve().parents[3]


def _trace_duration(trace: dict) -> float:
    groups = trace.get("frame_groups") or []
    if not groups:
        return 0.0
    return round(max(float(g["time_end"]) for g in groups), 3)


def ensure_clips(clips_dir: Path, make_fixtures: Callable[[Path], None]) -> None:
    """Regenerate the demo clips into `clips_dir` if any are missing."""
    if clips_dir.exists() and any(clips_dir.glob("*.mp4")):
        return
    clips_dir.mkdir(parents=True, exist_ok=True)
    make_fixtures(clips_dir)


def _default_make_fixtures(clips_dir: Path) -> None:
    # Imported lazily: scripts/ is not part of the installed package, so this
    # only works from a source checkout — which is exactly the demo use case.
    import importlib.util
    import sys

    script = repo_root() / "scripts" / "make_fixtures.py"
    spec = importlib.util.spec_from_file_location("jsr_make_fixtures", script)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for name, (seconds, gen) in mod.CLIPS.items():
        mod.write_clip(clips_dir / name, seconds, gen)


def iter_fixture_traces(traces_dir: Path) -> Iterable[tuple[str, dict]]:
    for path in sorted(traces_dir.glob("*.trace.json")):
        yield path.stem.replace(".trace", ""), json.loads(path.read_text(encoding="utf-8"))


def seed_demo(
    video_store: VideoStore,
    trace_store: TraceStore,
    *,
    root: Path | None = None,
    make_fixtures: Callable[[Path], None] | None = None,
) -> list[str]:
    """Seed both stores from the fixtures. Returns the seeded trace_ids.

    Idempotent: a trace already present (cache hit) is skipped, so re-running
    `jsr up --demo` is cheap.
    """
    root = root or repo_root()
    traces_dir = root / "fixtures" / "traces"
    clips_dir = root / "fixtures" / "clips"
    ensure_clips(clips_dir, make_fixtures or _default_make_fixtures)

    seeded: list[str] = []
    for name, trace in iter_fixture_traces(traces_dir):
        video_id = trace["video_id"]
        question = trace["question"]
        trace_id = trace_id_for(video_id, question)
        duration_s = _trace_duration(trace)

        clip = clips_dir / f"{name}.mp4"
        if clip.exists():
            video_store.save(video_id, f"{name}.mp4", clip.read_bytes(), duration_s)

        if not trace_store.has(trace_id):
            lib_item = {
                "trace_id": trace_id,
                "video_id": video_id,
                "question": question,
                "answer": trace.get("answer", ""),
                "created_at": "2026-01-01T00:00:00Z",  # stable ordering for the demo library
                "duration_s": duration_s,
            }
            trace_store.put(trace_id, trace, lib_item)
        seeded.append(trace_id)
    return seeded
