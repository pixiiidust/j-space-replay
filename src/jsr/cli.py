"""`jsr` console entry point (Milestone 5).

    jsr up                 # serve the API + built frontend on localhost:8000
    jsr up --demo          # pre-seed the bundled demo library, no GPU needed
    jsr up --port 9000     # pick a port

`up` builds the FastAPI app (jsr.server.api.create_app) and mounts the built
frontend (frontend/dist) at "/", with the API routes taking precedence. GPU
jobs still queue normally when a model is available; `--demo` additionally
seeds the three pre-baked demo traces so the library is instantly browsable
with zero GPU work.

Nothing here imports torch or touches CUDA at import time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jsr.server.api import create_app
from jsr.server.demo import repo_root, seed_demo


def _build_app(args: argparse.Namespace):
    app = create_app(uploads_dir=args.uploads_dir, traces_dir=args.traces_dir)

    if args.demo:
        try:
            seeded = seed_demo(app.state.video_store, app.state.trace_store)
            print(f"[jsr] demo mode: seeded {len(seeded)} bundled trace(s) — no GPU needed.")
        except Exception as exc:  # noqa: BLE001 - demo seeding is best-effort
            print(f"[jsr] warning: could not seed demo fixtures ({exc}).", file=sys.stderr)
    else:
        _startup_vram_notice()

    _mount_frontend(app, args.frontend_dist)
    return app


def _startup_vram_notice() -> None:
    """Non-fatal VRAM pre-flight at startup: warn, but still serve the UI."""
    from jsr.server.preflight import InsufficientVRAMError, check_vram

    try:
        free = check_vram()
        print(f"[jsr] GPU pre-flight OK: {free:.1f} GiB free VRAM.")
    except InsufficientVRAMError as exc:
        print(f"[jsr] GPU pre-flight: {exc}", file=sys.stderr)
        print("[jsr] Starting anyway; GPU trace jobs will be refused until this is "
              "resolved. Use `jsr up --demo` to browse the bundled traces.", file=sys.stderr)


def _mount_frontend(app, dist: Path) -> None:
    from fastapi.responses import PlainTextResponse
    from fastapi.staticfiles import StaticFiles

    if dist.exists() and (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
        print(f"[jsr] serving frontend from {dist}")
    else:
        msg = (
            "Frontend build not found at "
            f"{dist}. Build it first:\n\n"
            "    npm --prefix frontend ci\n"
            "    npm --prefix frontend run build\n\n"
            "The API is still available under /videos, /traces, /jobs, /library."
        )
        print(f"[jsr] warning: {msg}", file=sys.stderr)

        @app.get("/", response_class=PlainTextResponse)
        async def _needs_build() -> str:  # pragma: no cover - trivial fallback route
            return msg


def _cmd_up(args: argparse.Namespace) -> int:
    import uvicorn

    app = _build_app(args)
    print(f"[jsr] listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jsr", description="J-Space-Replay launcher")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="serve the backend + built frontend on localhost")
    up.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    up.add_argument("--port", type=int, default=8000, help="port (default 8000)")
    up.add_argument("--demo", action="store_true",
                    help="pre-seed the bundled demo library (no GPU required)")
    up.add_argument("--uploads-dir", default="uploads", help="where uploaded videos are stored")
    up.add_argument("--traces-dir", default="traces", help="where computed traces are stored")
    up.add_argument("--frontend-dist", type=Path, default=repo_root() / "frontend" / "dist",
                    help="path to the built frontend (frontend/dist)")
    up.set_defaults(func=_cmd_up)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
