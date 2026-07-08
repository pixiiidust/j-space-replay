"""`jsr up` CLI wiring (M5): arg parsing, frontend mount precedence, demo seed.

No server is actually run (uvicorn.run is never called); we build the app and
drive it with TestClient.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from jsr import cli


def _args(tmp_path, **over):
    ns = cli.build_parser().parse_args(["up"])
    ns.uploads_dir = str(tmp_path / "uploads")
    ns.traces_dir = str(tmp_path / "traces")
    ns.frontend_dist = tmp_path / "dist"
    ns.demo = False
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_parser_defaults_and_flags():
    ns = cli.build_parser().parse_args(["up", "--demo", "--port", "9001"])
    assert ns.command == "up"
    assert ns.demo is True
    assert ns.port == 9001
    assert ns.host == "127.0.0.1"


def test_up_requires_a_subcommand():
    import pytest

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_api_routes_take_precedence_over_static_mount(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>JSR-INDEX</html>", encoding="utf-8")

    app = cli._build_app(_args(tmp_path, frontend_dist=dist))
    with TestClient(app) as client:
        # API route wins
        assert client.get("/library").json() == {"items": []}
        # SPA index served at root
        root = client.get("/")
        assert root.status_code == 200
        assert "JSR-INDEX" in root.text


def test_missing_frontend_build_serves_helpful_message(tmp_path):
    app = cli._build_app(_args(tmp_path))  # dist does not exist
    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert "npm" in root.text and "run build" in root.text
        # API still works
        assert client.get("/library").status_code == 200


def test_demo_flag_seeds_library(tmp_path, monkeypatch):
    seen = {}

    def fake_seed(video_store, trace_store, **kw):
        seen["called"] = True
        return ["t1", "t2", "t3"]

    monkeypatch.setattr(cli, "seed_demo", fake_seed)
    app = cli._build_app(_args(tmp_path, demo=True))
    assert seen.get("called") is True
    assert app is not None
