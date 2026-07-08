"""VRAM pre-flight guard (M5) — driven entirely with fakes, no CUDA."""

from __future__ import annotations

import pytest

from jsr.server.preflight import (
    GIB,
    InsufficientVRAMError,
    check_vram,
    free_vram_gib,
    preflight_job,
)


def _mem(free_gib: float):
    return lambda: (int(free_gib * GIB), int(16 * GIB))


def test_free_vram_none_when_no_cuda():
    assert free_vram_gib(_mem(14), cuda_available=lambda: False) is None


def test_free_vram_reports_gib():
    got = free_vram_gib(_mem(13.5), cuda_available=lambda: True)
    assert got == pytest.approx(13.5, abs=0.01)


def test_check_vram_ok_above_threshold():
    free = check_vram(mem_get_info=_mem(14), cuda_available=lambda: True)
    assert free == pytest.approx(14, abs=0.01)


def test_check_vram_refuses_below_threshold_with_guidance():
    with pytest.raises(InsufficientVRAMError) as ei:
        check_vram(mem_get_info=_mem(6), cuda_available=lambda: True)
    assert "6.0 GiB" in str(ei.value)
    assert "demo" in str(ei.value).lower()


def test_check_vram_refuses_when_no_cuda_at_startup():
    with pytest.raises(InsufficientVRAMError) as ei:
        check_vram(mem_get_info=_mem(99), cuda_available=lambda: False)
    assert "No CUDA GPU" in str(ei.value)


def test_preflight_job_is_soft_when_no_cuda():
    # Soft guard: no GPU -> do nothing (model load / mocks handle it).
    assert preflight_job(cuda_available=lambda: False) is None


def test_preflight_job_refuses_when_measurably_low():
    with pytest.raises(InsufficientVRAMError):
        preflight_job(mem_get_info=_mem(4), cuda_available=lambda: True)


def test_preflight_job_ok_when_enough():
    assert preflight_job(mem_get_info=_mem(15), cuda_available=lambda: True) == pytest.approx(15, abs=0.01)
