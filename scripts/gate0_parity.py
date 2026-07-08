"""Gate 0: true-weight layer forward parity vs the NF4 runtime layer.

Rebuilds decoder layers with the ORIGINAL checkpoint weights (bf16 on disk,
run here in fp16 like the runtime) and replays real captured hidden states
from a video prefill through them. The measured drift is the NF4-vs-true
weight gap on a single layer forward — the same gap the J-lens tolerates when
it is fit on true weights but applied to NF4-runtime activations. The number
goes in the lens meta caveat.

A wiring bug (wrong mask/rope/config) would show as order-1 error; NF4 weight
quantization alone should show as a few-percent error on the branch delta
(out - in) and much less on the full output (residual stream dilutes it).

    uv run python scripts/gate0_parity.py [--layers 0 12 27] [--clip fixtures/clips/ball_drop.mp4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jsr.jweights import CheckpointWeights, build_true_layer  # noqa: E402
from jsr.model import decoder_layers, load_model_and_processor  # noqa: E402
from jsr.trace import prepare_video_inputs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 12, 27])
    ap.add_argument("--clip", default="fixtures/clips/ball_drop.mp4")
    ap.add_argument("--question", default="Why does the ball fall?")
    ap.add_argument("-o", "--out", default="reports/jlens_gate0.json")
    args = ap.parse_args()

    model, processor = load_model_and_processor()
    inputs, _, _ = prepare_video_inputs(processor, args.clip, args.question)
    layers = decoder_layers(model)

    captured: dict[int, dict] = {}

    def pre_hook(idx):
        def fn(_module, hook_args, hook_kwargs):
            hs = hook_args[0] if hook_args else hook_kwargs["hidden_states"]
            captured[idx] = {
                "input": hs.detach().clone(),
                "kwargs": {
                    k: v for k, v in hook_kwargs.items() if k != "hidden_states"
                },
            }
            return None

        return fn

    def post_hook(idx):
        def fn(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx]["output"] = hidden.detach().clone()
            return None

        return fn

    handles = []
    for i in args.layers:
        handles.append(layers[i].register_forward_pre_hook(pre_hook(i), with_kwargs=True))
        handles.append(layers[i].register_forward_hook(post_hook(i)))
    with torch.inference_mode():
        model(**inputs, use_cache=False)
    for h in handles:
        h.remove()

    def dequant_state_dict(layer) -> dict[str, torch.Tensor]:
        """The runtime layer's own weights, dequantized from Linear4bit."""
        import bitsandbytes.functional as bnbf
        from bitsandbytes.nn import Linear4bit

        sd = {}
        for name, mod in layer.named_modules():
            if isinstance(mod, Linear4bit):
                sd[f"{name}.weight"] = bnbf.dequantize_4bit(
                    mod.weight.data, mod.weight.quant_state
                ).to(torch.float16)
                if mod.bias is not None:
                    sd[f"{name}.bias"] = mod.bias.data.to(torch.float16)
            elif isinstance(mod, torch.nn.Linear):
                sd[f"{name}.weight"] = mod.weight.data.to(torch.float16)
                if mod.bias is not None:
                    sd[f"{name}.bias"] = mod.bias.data.to(torch.float16)
        for name, p in layer.named_parameters():
            if name not in sd and "weight" in name:
                sd[name] = p.data.to(torch.float16)
        return sd

    def replay(rebuilt, cap) -> dict[str, float]:
        with torch.inference_mode():
            out = rebuilt(cap["input"], **cap["kwargs"])
        got = (out[0] if isinstance(out, tuple) else out).float()
        ref, x = cap["output"].float(), cap["input"].float()
        delta_ref, delta_got = ref - x, got - x
        return {
            "rel_err_output": round(((got - ref).norm() / ref.norm()).item(), 6),
            "rel_err_branch_delta": round(
                ((delta_got - delta_ref).norm() / delta_ref.norm()).item(), 6
            ),
            "cos_branch_delta": round(
                torch.nn.functional.cosine_similarity(
                    delta_got.flatten(), delta_ref.flatten(), dim=0
                ).item(),
                6,
            ),
        }

    weights = CheckpointWeights()
    results = []
    for i in args.layers:
        cap = captured[i]
        # control: same wiring, runtime's own (dequantized) weights -> must be
        # near-exact; proves any true-weight gap below is quantization drift
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

        with torch.device("meta"):
            ctrl_layer = Qwen2_5_VLDecoderLayer(model.config.text_config, i)
        ctrl_layer.load_state_dict(dequant_state_dict(layers[i]), assign=True)
        ctrl_layer.eval()
        control = replay(ctrl_layer, cap)
        del ctrl_layer
        torch.cuda.empty_cache()

        true_layer = build_true_layer(model.config.text_config, i, weights, dtype=torch.float16)
        true = replay(true_layer, cap)
        del true_layer
        torch.cuda.empty_cache()

        row = {"layer": i, "seq": int(cap["input"].shape[1]),
               "control_dequant": control, "true_weights": true}
        results.append(row)
        print(row, flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "_note": "Gate 0 — true-weight (bf16 checkpoint, fp16 compute) layer forward "
                 "vs NF4 runtime layer, same captured input from a real video prefill. "
                 "rel_err_branch_delta isolates the transformed part (out - in).",
        "clip": args.clip,
        "question": args.question,
        "results": results,
    }, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
