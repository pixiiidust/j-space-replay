"""Forward-hook residual capture, streamed to CPU inside the hook.

One forward `model.generate(...)` drives both phases:
  - prefill: each layer fires once with (1, seq, d) — visual + text positions
  - decode:  each layer fires per generated token with (1, 1, d)

The hook copies to pinned-free CPU fp16 immediately and keeps NO reference to
the GPU tensor; `assert_no_gpu_accumulation` verifies that invariant.
"""

from __future__ import annotations

import torch


class ResidualCapture:
    def __init__(self, layers):
        self._layers = list(layers)
        self._handles = []
        self.prefill: list[torch.Tensor | None] = [None] * len(self._layers)  # (seq, d) cpu fp16
        self.steps: list[list[torch.Tensor]] = [[] for _ in self._layers]  # per-layer list of (d,)

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    def _hook(self, layer_idx: int):
        def fn(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            cpu = hidden.detach()[0].to("cpu", dtype=torch.float16)  # (seq, d), breaks GPU ref
            if cpu.shape[0] > 1:
                # prefill (or a re-prefill: keep the latest full-sequence pass)
                self.prefill[layer_idx] = cpu
            else:
                self.steps[layer_idx].append(cpu[0])
            return None

        return fn

    def __enter__(self):
        for i, layer in enumerate(self._layers):
            self._handles.append(layer.register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def assert_no_gpu_accumulation(self) -> None:
        for t in self.prefill:
            assert t is None or t.device.type == "cpu", "prefill residual left on GPU"
        for per_layer in self.steps:
            for t in per_layer:
                assert t.device.type == "cpu", "step residual left on GPU"

    def prefill_stack(self) -> torch.Tensor:
        """(n_layers, seq, d) cpu fp16."""
        assert all(t is not None for t in self.prefill), "no prefill captured"
        return torch.stack(self.prefill)  # type: ignore[arg-type]

    def steps_stack(self) -> torch.Tensor:
        """(n_layers, n_steps, d) cpu fp16; empty tensor if nothing generated."""
        if not self.steps[0]:
            return torch.empty(self.n_layers, 0, 0)
        return torch.stack([torch.stack(per_layer) for per_layer in self.steps])
