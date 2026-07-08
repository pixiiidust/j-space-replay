"""Logit-lens decode: unembed(final_norm(residual)) -> top-k vocab tokens.

This is lens "logit-lens-v1" — and also the identity first link of the future
J-lens chain (J = I), so nothing here is throwaway. Strengths are raw fp16
logits in M1; per-layer z-score normalization lands in M2.
"""

from __future__ import annotations

import torch

from jsr.model import final_norm, unembedding


@torch.inference_mode()
def lens_topk(
    model,
    residuals: torch.Tensor,
    k: int = 10,
    chunk: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """residuals: (N, d) on CPU fp16 -> (ids (N, k), logits (N, k)) on CPU.

    Batched through the GPU in chunks; only the top-k survives on GPU before
    offload, never the (N, vocab) logits.
    """
    norm = final_norm(model)
    head = unembedding(model)
    device = next(head.parameters()).device
    ids_out, val_out = [], []
    for start in range(0, residuals.shape[0], chunk):
        h = residuals[start : start + chunk].to(device, dtype=torch.float16)
        logits = head(norm(h))
        vals, ids = torch.topk(logits.float(), k, dim=-1)
        ids_out.append(ids.cpu())
        val_out.append(vals.cpu())
    return torch.cat(ids_out), torch.cat(val_out)


def decode_tokens(tokenizer, ids: torch.Tensor) -> list[list[str]]:
    """Token ids (N, k) -> readable token strings, one list per row."""
    return [
        [tokenizer.decode([tid]) for tid in row]
        for row in ids.tolist()
    ]
