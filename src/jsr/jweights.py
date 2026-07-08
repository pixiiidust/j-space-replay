"""True-precision decoder-layer weights from the cached HF checkpoint.

The runtime model is bitsandbytes NF4; the J-lens Jacobians are computed from
the ORIGINAL checkpoint weights (bf16 on disk) so the analytic math carries no
dequantization drift. Weights load one layer at a time straight from the
safetensors shards — never through bitsandbytes.

The remaining, deliberate gap: the lens is fit on true weights but applied to
NF4-runtime activations. Gate 0 (scripts/gate0_parity.py) measures that drift;
the number ships in the lens meta caveat.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from jsr.model import MODEL_ID


def snapshot_path(model_id: str = MODEL_ID) -> Path:
    """Local path of the cached checkpoint (no network)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, local_files_only=True))


class CheckpointWeights:
    """Lazy per-tensor reads from the original safetensors shards."""

    def __init__(self, model_id: str = MODEL_ID):
        self.snap = snapshot_path(model_id)
        index = json.loads((self.snap / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._handles: dict[str, object] = {}

    def tensor(self, name: str, device="cuda", dtype=torch.float32) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(self.snap / shard, framework="pt")
            self._handles[shard] = handle
        return handle.get_tensor(name).to(device, dtype=dtype)

    def layer_state_dict(
        self, layer_idx: int, device="cuda", dtype=torch.float16
    ) -> dict[str, torch.Tensor]:
        """All tensors of decoder layer `layer_idx`, keys relative to the layer."""
        prefix = f"model.layers.{layer_idx}."
        return {
            name[len(prefix):]: self.tensor(name, device, dtype)
            for name in self.weight_map
            if name.startswith(prefix)
        }


def text_config(model_id: str = MODEL_ID):
    """The text-decoder config straight from the cached checkpoint (SDPA)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(snapshot_path(model_id), local_files_only=True)
    cfg = cfg.text_config
    cfg._attn_implementation = "sdpa"
    return cfg


def build_true_layer(cfg, layer_idx: int, weights: CheckpointWeights, dtype=torch.float16):
    """Decoder layer `layer_idx` rebuilt with original checkpoint weights.

    `cfg` is the text config (model.config.text_config or jweights.text_config()).
    Uses the same transformers layer class and attn implementation as the
    runtime model, so a captured (hidden_states, kwargs) replays exactly.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

    with torch.device("meta"):
        layer = Qwen2_5_VLDecoderLayer(cfg, layer_idx)
    layer.load_state_dict(weights.layer_state_dict(layer_idx, dtype=dtype), assign=True)
    layer.eval()
    return layer
