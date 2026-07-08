"""Verify each analytic Jacobian component against autograd ground truth.

Ground truth is torch autograd through the TRUE-WEIGHT rebuilt layer (proven
bit-exact against the runtime by Gate 0), in fp32. Run gate-by-gate:

    uv run python scripts/verify_jacobians.py capture   # once: stash real activations
    uv run python scripts/verify_jacobians.py rmsnorm
    uv run python scripts/verify_jacobians.py mlp
    uv run python scripts/verify_jacobians.py attn
    uv run python scripts/verify_jacobians.py layer

`capture` runs a real video prefill (ball_drop) on the NF4 runtime model and
stashes every layer's input hidden state + rotary embeddings to jlens/
(gitignored). The component verifies then run from the stash + the checkpoint
weights — no runtime model load, fast to iterate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jsr.jacobian import SKIP_FIRST, mlp_branch_jacobian, rms_norm_jacobian, valid_mask  # noqa: E402
from jsr.jweights import CheckpointWeights, build_true_layer, text_config  # noqa: E402

STASH = Path("jlens/stash_ball_drop.pt")


def true_layer_fp32(layer_idx: int, weights: CheckpointWeights):
    return build_true_layer(text_config(), layer_idx, weights, dtype=torch.float32)


def rope_from_stash(stash) -> tuple[torch.Tensor, torch.Tensor]:
    return stash["cos"].to("cuda"), stash["sin"].to("cuda")


def h_mid_from_stash(layer, stash, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(h_in, h_mid): residual entering the layer and entering the MLP branch."""
    h = stash["h_in"][layer_idx].to("cuda", torch.float32)
    with torch.no_grad():
        r = layer.self_attn(
            layer.input_layernorm(h[None]),
            position_embeddings=rope_from_stash(stash),
            attention_mask=None,
        )[0][0]
    return h, h + r


def capture(clip: str, question: str) -> None:
    from jsr.model import decoder_layers, load_model_and_processor
    from jsr.trace import prepare_video_inputs

    model, processor = load_model_and_processor()
    inputs, _, _ = prepare_video_inputs(processor, clip, question)
    layers = decoder_layers(model)

    h_in: list[torch.Tensor | None] = [None] * len(layers)
    extras: dict = {}

    def pre_hook(idx):
        def fn(_m, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            h_in[idx] = hs.detach()[0].to("cpu", torch.float16)
            if idx == 0:
                cos, sin = kwargs["position_embeddings"]
                extras["cos"] = cos.detach().to("cpu", torch.float32)
                extras["sin"] = sin.detach().to("cpu", torch.float32)
                am = kwargs.get("attention_mask")
                extras["attention_mask_none"] = am is None
            return None

        return fn

    def final_hook(_m, _a, output):
        hidden = output[0] if isinstance(output, tuple) else output
        extras["h_final"] = hidden.detach()[0].to("cpu", torch.float16)
        return None

    handles = [
        layer.register_forward_pre_hook(pre_hook(i), with_kwargs=True)
        for i, layer in enumerate(layers)
    ]
    handles.append(layers[-1].register_forward_hook(final_hook))
    with torch.no_grad():
        model(**inputs, use_cache=False)
    for h in handles:
        h.remove()

    STASH.parent.mkdir(exist_ok=True)
    torch.save({"h_in": torch.stack(h_in), **extras, "clip": clip}, STASH)
    print(f"stashed {STASH}: h_in {torch.stack(h_in).shape}, "
          f"mask_none={extras['attention_mask_none']}")


def load_stash():
    assert STASH.exists(), f"run `verify_jacobians.py capture` first ({STASH} missing)"
    return torch.load(STASH, weights_only=True)


def report(name: str, analytic: torch.Tensor, ref: torch.Tensor) -> None:
    rel = ((analytic - ref).norm() / ref.norm()).item()
    ok = "PASS" if rel < 1e-3 else "FAIL"
    print(f"{name}: rel_fro_err={rel:.3e}  [{ok}]  "
          f"(|ref|={ref.norm().item():.3e}, |analytic|={analytic.norm().item():.3e})")


def verify_rmsnorm(layer_idx: int, n_pos: int = 32) -> None:
    """Closed form vs per-position jacrev of the true layer's input_layernorm."""
    stash = load_stash()
    weights = CheckpointWeights()
    w = weights.tensor(f"model.layers.{layer_idx}.input_layernorm.weight",
                       dtype=torch.float32)
    x_full = stash["h_in"][layer_idx].to("cuda", torch.float32)  # (S, D)
    S = x_full.shape[0]
    torch.manual_seed(0)
    vm = valid_mask(S)
    idx = vm.nonzero()[:, 0][torch.randperm(int(vm.sum()))[:n_pos]]
    x = x_full[idx]
    eps = 1e-6

    def f(v):
        return v / torch.sqrt((v * v).mean() + eps) * w

    ref = torch.stack([torch.func.jacrev(f)(x[i]) for i in range(len(idx))]).mean(0)
    analytic = rms_norm_jacobian(x, w, eps)
    report(f"rmsnorm layer {layer_idx} ({n_pos} real positions)", analytic, ref)


def verify_mlp(layer_idx: int, n_pos: int = 8) -> None:
    """Analytic MLP-branch Jacobian vs per-position jacrev of mlp(post_ln(.))."""
    stash = load_stash()
    weights = CheckpointWeights()
    layer = true_layer_fp32(layer_idx, weights)
    _, h_mid = h_mid_from_stash(layer, stash, layer_idx)
    S = h_mid.shape[0]
    torch.manual_seed(0)
    vm = valid_mask(S)
    idx = vm.nonzero()[:, 0][torch.randperm(int(vm.sum()))[:n_pos]]
    x = h_mid[idx].detach()

    def f(v):
        return layer.mlp(layer.post_attention_layernorm(v))

    ref = torch.stack([torch.func.jacrev(f)(x[i]) for i in range(len(idx))]).mean(0)
    analytic = mlp_branch_jacobian(layer, x, torch.ones(len(idx), device="cuda"))
    report(f"mlp branch layer {layer_idx} ({n_pos} real positions)", analytic, ref)


def verify_attn(layer_idx: int, s_trunc: int = 64, n_cotangents: int = 8) -> None:
    """Analytic attention-branch Jacobian vs autograd through the true layer's
    REAL attention path (SDPA, math backend so it vmaps).

    Check 1 — exact full (D, D) on the first `s_trunc` real positions.
    Check 2 — random W_o-space cotangent rows at full sequence length.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from jsr.jacobian import attn_branch_jacobian

    stash = load_stash()
    weights = CheckpointWeights()
    layer = true_layer_fp32(layer_idx, weights)
    cos_f, sin_f = rope_from_stash(stash)

    # -- check 1: truncated exact
    x = stash["h_in"][layer_idx][:s_trunc].to("cuda", torch.float32)
    cos, sin = cos_f[:, :, :s_trunc], sin_f[:, :, :s_trunc]
    vm = valid_mask(s_trunc)
    V = vm.sum()

    def branch_sum(x2d):
        out = layer.self_attn(
            layer.input_layernorm(x2d[None]),
            position_embeddings=(cos, sin), attention_mask=None,
        )[0][0]
        return (out * vm[:, None]).sum(0)  # (D,) sum over valid targets

    with sdpa_kernel(SDPBackend.MATH):
        j_full = torch.func.jacrev(branch_sum, chunk_size=256)(x)  # (D, S', D)
    ref = torch.einsum("dsk,s->dk", j_full, vm) / V
    del j_full
    analytic = attn_branch_jacobian(layer, x, (cos, sin), vm)
    report(f"attn branch layer {layer_idx} (exact, S={s_trunc})", analytic, ref)

    # -- check 2: random cotangent rows at full length
    x = stash["h_in"][layer_idx].to("cuda", torch.float32)
    S = x.shape[0]
    vm = valid_mask(S)
    V = vm.sum()

    def branch_full(x2d):
        return layer.self_attn(
            layer.input_layernorm(x2d[None]),
            position_embeddings=(cos_f, sin_f), attention_mask=None,
        )[0][0]

    torch.manual_seed(1)
    u = torch.nn.functional.normalize(torch.randn(n_cotangents, x.shape[1], device="cuda"), dim=-1)
    with sdpa_kernel(SDPBackend.MATH):
        _, vjp_fn = torch.func.vjp(branch_full, x)
        rows_ref = []
        for r in range(n_cotangents):
            (g,) = vjp_fn(vm[:, None] * u[r][None, :])  # (S, D)
            rows_ref.append(vm @ g / V)
    rows_ref = torch.stack(rows_ref)
    analytic = attn_branch_jacobian(layer, x, (cos_f, sin_f), vm)
    report(f"attn branch layer {layer_idx} (random rows, S={S})", u @ analytic, rows_ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("component", choices=["capture", "rmsnorm", "mlp", "attn", "layer"])
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--clip", default="fixtures/clips/ball_drop.mp4")
    ap.add_argument("--question", default="Why does the ball fall?")
    args = ap.parse_args()

    if args.component == "capture":
        capture(args.clip, args.question)
    elif args.component == "rmsnorm":
        for li in (0, args.layer, 27):
            verify_rmsnorm(li)
    elif args.component == "mlp":
        for li in (0, args.layer, 27):
            verify_mlp(li)
    elif args.component == "attn":
        for li in (0, args.layer, 27):
            verify_attn(li)
    else:
        raise SystemExit(f"{args.component}: not implemented yet (gate-by-gate)")


if __name__ == "__main__":
    main()
