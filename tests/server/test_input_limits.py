"""POST /videos input limits (M5): size 413, duration 422, undecodable 415.

All hermetic — a fake probe_fn stands in for container probing so nothing
touches ffmpeg or the model.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from jsr.server.api import create_app


def _client(tmp_path, *, probe, max_upload_bytes=100 * 1024 * 1024, max_duration_s=25.0):
    app = create_app(
        uploads_dir=tmp_path / "uploads",
        traces_dir=tmp_path / "traces",
        probe_fn=probe,
        max_upload_bytes=max_upload_bytes,
        max_duration_s=max_duration_s,
    )
    return TestClient(app)


def _upload(client, data: bytes, name="clip.mp4"):
    return client.post("/videos", files={"file": (name, data, "video/mp4")})


def test_oversize_upload_rejected_413(tmp_path):
    with _client(tmp_path, probe=lambda d, f: 10.0, max_upload_bytes=16) as client:
        r = _upload(client, b"x" * 64)
        assert r.status_code == 413
        assert "limit" in r.json()["detail"].lower()


def test_too_long_clip_rejected_422(tmp_path):
    with _client(tmp_path, probe=lambda d, f: 99.0) as client:
        r = _upload(client, b"a-real-enough-clip")
        assert r.status_code == 422
        detail = r.json()["detail"].lower()
        assert "99" in detail and "shorter" in detail


def test_undecodable_upload_rejected_415(tmp_path):
    # probe returns 0.0 == duration unreadable (corrupt / unsupported codec)
    with _client(tmp_path, probe=lambda d, f: 0.0) as client:
        r = _upload(client, b"not-a-real-video")
        assert r.status_code == 415
        assert "codec" in r.json()["detail"].lower()


def test_empty_upload_rejected_400(tmp_path):
    with _client(tmp_path, probe=lambda d, f: 10.0) as client:
        r = _upload(client, b"")
        assert r.status_code == 400


def test_valid_upload_within_limits_accepted(tmp_path):
    with _client(tmp_path, probe=lambda d, f: 12.5) as client:
        r = _upload(client, b"ok-bytes")
        assert r.status_code == 200
        assert r.json()["duration_s"] == 12.5
