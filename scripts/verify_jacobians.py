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

from jsr.jacobian import SKIP_FIRST, rms_norm_jacobian, valid_mask  # noqa: E402
from jsr.jweights import CheckpointWeights, build_true_layer, text_config  # noqa: E402

STASH = Path("jlens/stash_ball_drop.pt")


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
    else:
        raise SystemExit(f"{args.component}: not implemented yet (gate-by-gate)")


if __name__ == "__main__":
    main()
