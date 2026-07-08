"""Analytic per-layer Jacobians for the J-lens.

Attribution: the assembly pattern — closed-form RMSNorm Jacobian (diag +
rank-1), SwiGLU "Hadamard trick", W_o-row cotangents through the softmax
attention core, per-position input-norm folding before one stacked GEMM —
is ported from WeZZard/jlens-qwen36 (Apache 2.0; see NOTICE). None of its
MLX/Metal/GDN code transfers; Qwen2.5-VL is 100% standard softmax attention.

Position convention (theirs, kept): every Jacobian is averaged over the valid
positions [skip_first, S-1):

    M[c, d] = (1/V) * sum_{t valid} sum_{s valid} d(out_t[c]) / d(x_s[d])

Attention contributes cross-position (t != s) terms; norm/MLP terms are
position-diagonal. Chaining averaged factors (J_{l-1} = J_l @ M_l) is the
lens's stated position-averaging approximation: E[prod M] ~ prod E[M].

All math is fp32 on true (bf16-checkpoint) weights — never on NF4 runtime
weights; Gate 0 (reports/jlens_gate0.json) measures that runtime drift.
"""

from __future__ import annotations

import torch

SKIP_FIRST = 4


def valid_mask(S: int, skip_first: int = SKIP_FIRST, device="cuda") -> torch.Tensor:
    """1.0 at positions [skip_first, S-1), 0.0 elsewhere (sinks + last)."""
    ar = torch.arange(S, device=device)
    return ((ar >= skip_first) & (ar < S - 1)).float()


def rms_norm_jacobian(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Closed-form Jacobian of RMSNorm y = x / rms(x) * w, position-averaged.

    Per position s:  dy/dx = diag(w)/r_s - (w .* x_hat_s) x_hat_s^T / (D r_s)
    with r_s = sqrt(mean(x_s^2) + eps), x_hat_s = x_s / r_s.

    x: (S, D); weight: (D,); mask: (S,) validity weights. Returns (D, D) fp32.
    """
    S, D = x.shape
    xf = x.float()
    r = torch.sqrt((xf * xf).mean(dim=-1, keepdim=True) + eps)  # (S, 1)
    x_hat = xf / r
    w = weight.float()
    m = torch.ones(S, device=x.device) if mask is None else mask.float()
    n_valid = m.sum()

    m_over_r = m / r[:, 0]  # (S,)
    diag = w * (m_over_r.sum() / n_valid)  # (D,)
    wx_hat = w * x_hat  # (S, D)
    rank = (m_over_r[:, None] * wx_hat).T @ x_hat / (n_valid * D)  # (D, D)
    return torch.diag(diag) - rank
