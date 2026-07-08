"""Lens decode: unembed(final_norm(residual)) -> top-k vocab tokens.

Two lenses share this path:
  - "logit-lens-v1" (default): the raw residual is read directly — the
    identity link of the J-lens chain.
  - "j-lens-v1": the residual is first transported into final-residual space
    by the fitted per-layer Jacobian, J_l @ h, then read the same way
    (softmax(W_U . norm(J_l h))). Fit by scripts/fit_jlens.py; see
    jsr/jacobian.py for the math and its stated approximations.

Strengths are raw fp16 logits (answer path) / patch shares (visual path);
per-layer z-score normalization happens in labels.py against baseline stats.
"""

from __future__ import annotations

from pathlib import Path

import torch

from jsr.model import final_norm, unembedding

JLENS_PATH = Path("jlens/j_lens_v1.pt")


class JLens:
    """Fitted analytic J-lens: per-layer transport into final-residual space."""

    def __init__(self, J: torch.Tensor, meta: dict, device="cuda"):
        # fp16 carries the lens's full information content (fit noise ~1e-2);
        # halves VRAM (28 x 3584^2 -> 686 MB resident while tracing)
        self.J = J.to(device, torch.float16)
        self.meta = meta

    @classmethod
    def load(cls, path: str | Path = JLENS_PATH, device="cuda") -> "JLens":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - the J-lens ships as a fit recipe, not weights; "
                "run `uv run python scripts/fit_jlens.py` (regenerates it, <2h on a "
                "16 GB GPU) or use the default logit lens"
            )
        data = torch.load(path, weights_only=True)
        return cls(data["J"], data["meta"], device=device)

    def transport(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """(N, d) residuals at `layer` -> (N, d) in final-residual space."""
        return h @ self.J[layer].T.to(h.device, h.dtype)


@torch.inference_mode()
def lens_topk(
    model,
    residuals: torch.Tensor,
    k: int = 10,
    chunk: int = 512,
    transport=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """residuals: (N, d) on CPU fp16 -> (ids (N, k), logits (N, k)) on CPU.

    Batched through the GPU in chunks; only the top-k survives on GPU before
    offload, never the (N, vocab) logits. `transport` (optional) maps each
    GPU chunk before the norm — the J-lens application point.
    """
    norm = final_norm(model)
    head = unembedding(model)
    device = next(head.parameters()).device
    ids_out, val_out = [], []
    for start in range(0, residuals.shape[0], chunk):
        h = residuals[start : start + chunk].to(device, dtype=torch.float16)
        if transport is not None:
            h = transport(h)
        logits = head(norm(h))
        vals, ids = torch.topk(logits.float(), k, dim=-1)
        ids_out.append(ids.cpu())
        val_out.append(vals.cpu())
    return torch.cat(ids_out), torch.cat(val_out)


@torch.inference_mode()
def lens_topk_layers(
    model,
    residuals: torch.Tensor,
    k: int = 10,
    chunk: int = 512,
    jlens: JLens | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer decode: residuals (L, N, d) -> (ids, logits) each (L, N, k).

    With a JLens, layer l's residuals are transported by J_l before the norm;
    without one this is the plain logit lens.
    """
    ids_out, val_out = [], []
    for layer in range(residuals.shape[0]):
        transport = None
        if jlens is not None:
            transport = lambda h, layer=layer: jlens.transport(h, layer)  # noqa: E731
        ids, vals = lens_topk(model, residuals[layer], k=k, chunk=chunk, transport=transport)
        ids_out.append(ids)
        val_out.append(vals)
    return torch.stack(ids_out), torch.stack(val_out)


def decode_tokens(tokenizer, ids: torch.Tensor) -> list[list[str]]:
    """Token ids (N, k) -> readable token strings, one list per row."""
    return [
        [tokenizer.decode([tid]) for tid in row]
        for row in ids.tolist()
    ]
