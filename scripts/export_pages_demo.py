"""Export a static, read-only demo site for GitHub Pages.

Builds the frontend in static-API mode and lays out pre-baked traces as
plain files with the SAME paths the live API serves (extensionless JSON;
videos as .mp4). Nothing dynamic survives: no uploads, no re-ask — the
library, replay, word grid, drills, pulses and unspoken strip all work,
because they are client-side over the trace JSON.

By default only SYNTHETIC clips are exported (fixtures + any --trace-id you
pass explicitly). Store traces of personal uploads are opt-in on purpose:
exporting them publishes the videos.

    uv run python scripts/export_pages_demo.py --base /j-space-replay
    uv run python scripts/export_pages_demo.py --trace-id 55338097ca187621-f27538f5-jl
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent


def group_duration(trace: dict) -> float:
    return max((g["time_end"] for g in trace["frame_groups"]), default=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    ap.add_argument("--base", default="/j-space-replay", help="URL prefix Pages serves under")
    ap.add_argument("--trace-id", nargs="*", default=[],
                    help="extra trace ids from the local traces/ store (publishes their videos!)")
    args = ap.parse_args()

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "traces").mkdir(parents=True)
    (out / "videos").mkdir()

    items: list[dict] = []
    lenses_seen: set[str] = set()

    def add(trace: dict, trace_id: str, lib: dict, video_src: Path) -> None:
        (out / "traces" / trace_id).write_text(
            json.dumps(trace, ensure_ascii=False), encoding="utf-8")
        vid = trace["video_id"]
        dst = out / "videos" / f"{vid}.mp4"
        if not dst.exists():
            shutil.copyfile(video_src, dst)
        items.append(lib)
        lenses_seen.add(trace.get("meta", {}).get("lens", "logit-lens-v1"))

    # committed fixture traces (synthetic clips; regenerate via make_fixtures)
    for tf in sorted((ROOT / "fixtures/traces").glob("*.trace.json")):
        trace = json.loads(tf.read_text(encoding="utf-8"))
        vid = trace["video_id"]
        clip = ROOT / "fixtures/clips" / f"{vid}.mp4"
        if not clip.exists():
            print(f"skip {tf.name}: no clip at {clip} (run scripts/make_fixtures.py)")
            continue
        trace_id = f"{vid}-demo"
        add(trace, trace_id, {
            "trace_id": trace_id, "video_id": vid, "question": trace["question"],
            "lens": trace["meta"].get("lens", "logit-lens-v1"),
            "answer": trace.get("answer", ""), "created_at": "",
            "duration_s": group_duration(trace),
        }, clip)

    # opt-in traces from the local server store
    for tid in args.trace_id:
        tpath = ROOT / "traces" / f"{tid}.json"
        lpath = ROOT / "traces" / f"{tid}.lib.json"
        trace = json.loads(tpath.read_text(encoding="utf-8"))
        lib = json.loads(lpath.read_text(encoding="utf-8"))
        vid = trace["video_id"]
        src = next(p for p in (ROOT / "uploads").glob(f"{vid}.*") if p.suffix != ".json")
        add(trace, tid, lib, src)

    (out / "library").write_text(
        json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
    (out / "lenses").write_text(
        json.dumps({"lenses": sorted(lenses_seen), "default": "logit-lens-v1"}),
        encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")

    # frontend in static-API mode, served under the Pages base path
    env = {"VITE_STATIC_API": "1", "VITE_API_BASE": args.base,
           "VITE_BASE": args.base.rstrip("/") + "/"}
    import os
    subprocess.run(["npm", "--prefix", "frontend", "run", "build"],
                   cwd=ROOT, check=True, shell=sys.platform == "win32",
                   env={**os.environ, **env})
    for p in (ROOT / "frontend/dist").iterdir():
        dest = out / p.name
        if p.is_dir():
            shutil.copytree(p, dest)
        else:
            shutil.copyfile(p, dest)

    n_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2**20
    print(f"exported {len(items)} traces to {out} ({n_mb:.1f} MB)")


if __name__ == "__main__":
    main()
