"""Frame sampler + frame-group bookkeeping (pure logic where possible)."""

from pathlib import Path

import pytest

from jsr.video import FrameGroup, group_frames, sample_frames

CLIPS = Path("fixtures/clips")


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures():
    if not (CLIPS / "ball_drop.mp4").exists():
        import subprocess
        import sys

        subprocess.run([sys.executable, "scripts/make_fixtures.py"], check=True)


def test_sample_1fps_counts():
    s = sample_frames(CLIPS / "ball_drop.mp4", fps=1.0)  # 8 s @ 24 fps
    assert len(s.frames) == 8
    assert s.timestamps == pytest.approx(list(range(8)), abs=0.05)
    assert s.duration == pytest.approx(8.0, abs=0.1)


def test_group_frames_two_frame_merge():
    s = sample_frames(CLIPS / "traffic.mp4", fps=1.0)  # 12 s -> 12 frames -> 6 groups
    groups = group_frames(s)
    assert len(groups) == 6
    g0 = groups[0]
    assert isinstance(g0, FrameGroup)
    assert g0.frame_indices == [0, 1]
    assert g0.time_start == pytest.approx(0.0, abs=0.05)
    # a group's window ends where the next begins
    assert g0.time_end == pytest.approx(groups[1].time_start, abs=0.05)
    assert groups[-1].frame_indices == [10, 11]


def test_group_frames_odd_count_pads_last():
    s = sample_frames(CLIPS / "ball_drop.mp4", fps=0.5)  # 8 s -> 4 frames? no: 0,2,4,6 -> 4
    # force odd by dropping one
    s.frames, s.timestamps = s.frames[:3], s.timestamps[:3]
    groups = group_frames(s)
    assert len(groups) == 2
    assert groups[-1].frame_indices == [2, 2]  # last frame repeated, matches Qwen pad
