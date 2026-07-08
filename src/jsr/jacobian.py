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


def mlp_branch_jacobian(
    layer, h_mid: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Jacobian of the MLP branch d(mlp(post_ln(h)))/dh, position-averaged.

    Exact within the branch: the post-attention RMSNorm is folded PER-POSITION
    (not as a product of position averages). Per position s:

        J(s) = W_down [diag(silu'(g_s) u_s) W_gate + diag(silu(g_s)) W_up] J_n(s)

    Splitting J_n(s) into its diag and rank-1 parts, the position sum becomes
    diagonal rescales of W_gate/W_up plus rank-|valid| corrections — a few
    GEMMs total (the Hadamard trick).

    layer: true-weight fp32 decoder layer; h_mid: (S, D) pre-norm residual
    entering the branch; mask: (S,) validity. Returns (D, D) fp32.
    """
    S, D = h_mid.shape
    w = layer.post_attention_layernorm.weight.detach().float()
    eps = layer.post_attention_layernorm.variance_epsilon
    W_gate = layer.mlp.gate_proj.weight.detach().float()  # (I, D)
    W_up = layer.mlp.up_proj.weight.detach().float()      # (I, D)
    W_down = layer.mlp.down_proj.weight.detach().float()  # (D, I)

    xf = h_mid.float()
    r = torch.sqrt((xf * xf).mean(dim=-1, keepdim=True) + eps)  # (S, 1)
    x_hat = xf / r
    xn = x_hat * w  # post_ln output, (S, D)
    m_over_r = mask.float() / r[:, 0]  # (S,)
    n_valid = mask.sum()

    g = xn @ W_gate.T  # (S, I)
    u = xn @ W_up.T    # (S, I)
    sig = torch.sigmoid(g)
    dA = sig * (1.0 + g * (1.0 - sig)) * u  # silu'(g) * u
    dB = g * sig                            # silu(g)

    wx_hat = w * x_hat  # (S, D)
    inner = torch.zeros(W_gate.shape[0], D, device=h_mid.device)
    for dcoef, W in ((dA, W_gate), (dB, W_up)):
        d1 = torch.einsum("si,s->i", dcoef, m_over_r)  # (I,)
        inner += (W * d1[:, None]) * w[None]
        p = wx_hat @ W.T  # (S, I)
        P = dcoef * p * (m_over_r / D)[:, None]  # (S, I)
        inner -= P.T @ x_hat
    return (W_down @ inner) / n_valid


def _fold_norm_project(
    draw: torch.Tensor, W_stack: torch.Tensor, x: torch.Tensor,
    w_norm: torch.Tensor, eps: float, mask: torch.Tensor,
) -> torch.Tensor:
    """Backward through the stacked input projections with the input RMSNorm
    folded per-position, position sum taken BEFORE the single GEMM.

    Uses g^T J_n(s) = (g .* w)/r_s - (g . (w .* x_hat_s)) x_hat_s^T / (D r_s).

    draw: (C, S, F) grads w.r.t. stacked projection outputs; W_stack: (F, D);
    x: (S, D) pre-norm residual. Returns (C, D) rows of the branch Jacobian.
    """
    S, D = x.shape
    xf = x.float()
    r = torch.sqrt((xf * xf).mean(dim=-1, keepdim=True) + eps)  # (S, 1)
    x_hat = xf / r
    w = w_norm.float()
    m_over_r = mask.float() / r[:, 0]  # (S,)
    n_valid = mask.sum()

    A = torch.einsum("csf,s->cf", draw, m_over_r)  # (C, F)
    term1 = (A @ W_stack) * w[None]  # (C, D)
    U = (w[None] * x_hat) @ W_stack.T  # (S, F)
    alpha = torch.einsum("csf,sf->cs", draw, U * m_over_r[:, None])  # (C, S)
    term2 = (alpha @ x_hat) / D  # (C, D)
    return (term1 - term2) / n_valid


def attn_branch_jacobian(
    layer, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor],
    mask: torch.Tensor, chunk: int = 16,
) -> torch.Tensor:
    """Jacobian of the attention branch d(self_attn(ln_in(x)))/dx, averaged
    over valid target AND source positions (cross-position terms included).

    Exact within the branch: cotangents are rows of W_o seeded at every valid
    target position, backpropagated in chunks through the softmax core (mRoPE +
    GQA + causal; primals replicated per chunk — vmapped backward blew past
    VRAM here), then the input RMSNorm is folded per-position before a single
    stacked-projection GEMM.

    layer: true-weight fp32 decoder layer; x: (S, D) pre-norm residual;
    rope: (cos, sin) as passed to the runtime layers. Returns (D, D) fp32.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
    )

    attn = layer.self_attn
    S, D = x.shape
    H, hd = attn.num_heads, attn.head_dim
    KV, rf = attn.num_key_value_heads, attn.num_key_value_groups
    scale = attn.scaling
    mrope_section = attn.config.rope_parameters["mrope_section"]
    cos, sin = (t.float() for t in rope)

    w_in = layer.input_layernorm.weight.detach().float()
    eps_in = layer.input_layernorm.variance_epsilon
    xf = x.float()
    xn = xf / torch.sqrt((xf * xf).mean(dim=-1, keepdim=True) + eps_in) * w_in

    W_q, b_q = attn.q_proj.weight.detach().float(), attn.q_proj.bias.detach().float()
    W_k, b_k = attn.k_proj.weight.detach().float(), attn.k_proj.bias.detach().float()
    W_v, b_v = attn.v_proj.weight.detach().float(), attn.v_proj.bias.detach().float()
    W_o = attn.o_proj.weight.detach().float()  # (D, H*hd)

    q_full = xn @ W_q.T + b_q  # (S, H*hd)
    k_pre = xn @ W_k.T + b_k   # (S, KV*hd)
    v_pre = xn @ W_v.T + b_v   # (S, KV*hd)

    ar = torch.arange(S, device=x.device)
    causal = torch.where(ar[None, :] <= ar[:, None], 0.0, -1e9).float()

    def core(qf, kp, vp):
        # qf: (C, S, H*hd), kp/vp: (C, S, KV*hd) -> (C, S, H*hd)
        C = qf.shape[0]
        q = qf.view(C, S, H, hd).transpose(1, 2)   # (C, H, S, hd)
        k = kp.view(C, S, KV, hd).transpose(1, 2)
        v = vp.view(C, S, KV, hd).transpose(1, 2)
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
        k = torch.repeat_interleave(k, rf, dim=1)
        v = torch.repeat_interleave(v, rf, dim=1)
        scores = q @ k.transpose(-1, -2) * scale + causal
        out = torch.softmax(scores, dim=-1) @ v    # (C, H, S, hd)
        return out.transpose(1, 2).reshape(C, S, H * hd)

    W_stack = torch.cat([W_q, W_k, W_v], dim=0)  # (F, D)
    M = torch.zeros(D, D, device=x.device)
    for c0 in range(0, D, chunk):
        rows_w = W_o[c0 : c0 + chunk]  # (C, H*hd)
        C = rows_w.shape[0]
        dy = mask[None, :, None] * rows_w[:, None, :]  # (C, S, H*hd)
        primals = [
            t[None].expand(C, *t.shape).clone().requires_grad_()
            for t in (q_full, k_pre, v_pre)
        ]
        out = core(*primals)
        grads = torch.autograd.grad(out, primals, dy)
        draw = torch.cat(grads, dim=-1)  # (C, S, F)
        M[c0 : c0 + C] = _fold_norm_project(draw, W_stack, x, w_in, eps_in, mask)
        del primals, out, dy, grads, draw
    return M


def decoder_layer_jacobian(
    layer, h_in: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor],
    mask: torch.Tensor, chunk: int = 16,
) -> torch.Tensor:
    """Full per-layer Jacobian M_l = d(h_out)/d(h_in), position-averaged.

        r = attn(ln_in(x));  h_mid = x + r;  out = h_mid + mlp(ln_post(h_mid))
        => M = M_mid + M_mlp_branch @ M_mid,  M_mid = I + M_attn_branch

    Both branch Jacobians are exact within their branch; the single remaining
    within-layer approximation is the averaged product junction M_mlp @ M_mid
    (position decorrelation, ~1e-2 rel per the reference's measurements).

    layer: true-weight fp32 decoder layer; h_in: (S, D) residual entering it.
    """
    S, D = h_in.shape
    x = h_in.float()
    M_attn = attn_branch_jacobian(layer, x, rope, mask, chunk=chunk)
    with torch.no_grad():
        r = layer.self_attn(
            layer.input_layernorm(x[None]),
            position_embeddings=rope, attention_mask=None,
        )[0][0]
    h_mid = x + r
    M_mlp = mlp_branch_jacobian(layer, h_mid, mask)
    M_mid = M_attn
    M_mid.diagonal().add_(1.0)  # I + M_attn, in place
    return M_mid + M_mlp @ M_mid
