"""VRAM pre-flight guard (Milestone 5).

A GPU trace pass on Qwen2.5-VL-7B (4-bit weights ~5.5 GB + KV cache + capture
buffers) needs headroom. Below ~12 GiB free the run either OOMs mid-pass or is
forced to caps so low the trace is useless, so we refuse up front with guidance
instead of failing deep inside a job.

Design rules (from the M5 brief):
  * NEVER touch CUDA at import time. `torch.cuda.mem_get_info` is imported and
    called only inside `free_vram_gib`, and only when CUDA is actually present.
  * Fully injectable/mockable: every function takes an optional `mem_get_info`
    (and `cuda_available`) so tests drive it without a GPU.
  * Demo mode never calls this — `jsr up --demo` serves pre-baked traces with
    no GPU work at all.
"""

from __future__ import annotations

from typing import Callable

GIB = 1024**3
MIN_FREE_GIB = 12.0


class InsufficientVRAMError(RuntimeError):
    """Raised at job time when free VRAM is below the safe threshold."""


def _default_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch missing / broken install -> treat as no CUDA
        return False


def _default_mem_get_info() -> tuple[int, int]:
    import torch

    # (free_bytes, total_bytes) for the current device
    return torch.cuda.mem_get_info()


def free_vram_gib(
    mem_get_info: Callable[[], tuple[int, int]] | None = None,
    *,
    cuda_available: Callable[[], bool] | None = None,
) -> float | None:
    """Free VRAM on the active CUDA device in GiB, or None if CUDA is absent.

    None means "no GPU here" — the caller decides what that implies (a bare
    `jsr up` with no CUDA still serves the demo library; a GPU job cannot run).
    """
    is_available = cuda_available or _default_cuda_available
    if not is_available():
        return None
    query = mem_get_info or _default_mem_get_info
    free_bytes, _total = query()
    return free_bytes / GIB


def guidance_text(free: float | None, *, min_gib: float = MIN_FREE_GIB) -> str:
    """Human-facing guidance shown when a GPU job is refused."""
    if free is None:
        return (
            "No CUDA GPU was detected. J-Space-Replay needs an NVIDIA GPU with "
            f"at least {min_gib:.0f} GiB of free VRAM to compute a trace. You can "
            "still explore the bundled demo traces with `jsr up --demo` (no GPU "
            "required)."
        )
    return (
        f"Only {free:.1f} GiB of GPU memory is free, but a trace pass needs "
        f"about {min_gib:.0f} GiB (4-bit weights ~5.5 GB plus KV cache and "
        "activation-capture buffers). Close other GPU programs (browsers, other "
        "model processes) and retry, or explore the bundled demo traces with "
        "`jsr up --demo`, which needs no GPU."
    )


def check_vram(
    *,
    min_gib: float = MIN_FREE_GIB,
    mem_get_info: Callable[[], tuple[int, int]] | None = None,
    cuda_available: Callable[[], bool] | None = None,
) -> float | None:
    """Strict guard for `jsr up` startup: raise if a GPU job could not proceed.

    Returns the free GiB on success (for logging). Raises when CUDA is present
    but below `min_gib` free, OR when CUDA is entirely absent (no GPU means no
    GPU job). Used at `jsr up` startup to tell a GPU-less user to use `--demo`.
    Callers that want demo-only behaviour simply never call this.
    """
    free = free_vram_gib(mem_get_info, cuda_available=cuda_available)
    if free is None:
        raise InsufficientVRAMError(guidance_text(None, min_gib=min_gib))
    if free < min_gib:
        raise InsufficientVRAMError(guidance_text(free, min_gib=min_gib))
    return free


def preflight_job(
    *,
    min_gib: float = MIN_FREE_GIB,
    mem_get_info: Callable[[], tuple[int, int]] | None = None,
    cuda_available: Callable[[], bool] | None = None,
) -> float | None:
    """Soft guard run just before a GPU job loads the model.

    Refuses ONLY when free VRAM is measurable and below `min_gib`. When CUDA is
    absent (None) it does nothing — the model load then fails with its own clear
    error, and this keeps CPU test/CI environments (and injected fakes) working.
    Returns the free GiB when measured, else None.
    """
    free = free_vram_gib(mem_get_info, cuda_available=cuda_available)
    if free is not None and free < min_gib:
        raise InsufficientVRAMError(guidance_text(free, min_gib=min_gib))
    return free
