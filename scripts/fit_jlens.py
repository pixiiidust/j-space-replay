"""Fit the analytic J-lens over a video-prompt corpus.

Two phases:
  A) capture — load the NF4 runtime model once, prefill every corpus clip
     (video + default question), stash per-layer inputs + rotary embeddings
     to jlens/acts/ (fp16, ~170 MB/clip).
  B) fit — model unloaded. Chain top-down with true-weight fp32 layers:
     J_27 = J_finalnorm;  J_{l-1} = J_l @ M_l  (M_l at the layer's real input)
     accumulated per clip, layer-major so each layer's weights load once.
     Averaging J over clips is the lens's stated position/prompt-averaging
     approximation.

Output: jlens/j_lens_v1.pt — (28, D, D) fp32 + honesty meta. NOT committed
(jlens/ is gitignored); this script regenerates it. Checkpointed every 4
layers; safe to re-run after an interruption.

    uv run python scripts/fit_jlens.py            # both phases, resume-aware
    uv run python scripts/fit_jlens.py --capture-only
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jsr.jacobian import decoder_layer_jacobian, rms_norm_jacobian, valid_mask  # noqa: E402
from jsr.jweights import CheckpointWeights, build_true_layer, text_config  # noqa: E402

ACTS = Path("jlens/acts")
CKPT = Path("jlens/fit_ckpt.pt")


def default_corpus() -> list[Path]:
    clips = Path("fixtures/clips")
    corpus = [clips / n for n in ("ball_drop.mp4", "shape_morph.mp4", "traffic.mp4")]
    corpus += sorted(clips.glob("gate_*.mp4"))
    corpus += sorted((clips / "baseline").glob("baseline_*.mp4"))[::4]
    return [c for c in corpus if c.exists()]


def capture_corpus(corpus: list[Path]) -> None:
    from jsr.model import decoder_layers, load_model_and_processor
    from jsr.trace import prepare_video_inputs

    todo = [c for c in corpus if not (ACTS / f"{c.stem}.pt").exists()]
    if not todo:
        print("capture: all stashes present")
        return
    ACTS.mkdir(parents=True, exist_ok=True)
    model, processor = load_model_and_processor()
    layers = decoder_layers(model)
    for clip in todo:
        inputs, _, _ = prepare_video_inputs(processor, clip)
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
        torch.save({"h_in": torch.stack(h_in), **extras}, ACTS / f"{clip.stem}.pt")
        print(f"captured {clip.stem}: S={h_in[0].shape[0]}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def fit(corpus: list[Path], out_path: Path, chunk: int, resume: bool) -> None:
    weights = CheckpointWeights()
    cfg = text_config()
    n_layers, eps = cfg.num_hidden_layers, cfg.rms_norm_eps
    D = cfg.hidden_size
    stashes = [torch.load(ACTS / f"{c.stem}.pt", weights_only=True) for c in corpus]

    if resume and CKPT.exists():
        state = torch.load(CKPT, weights_only=True)
        assert state["corpus"] == [c.stem for c in corpus], "checkpoint corpus mismatch"
        next_l = state["next_l"]
        J_sum = state["J_sum"]
        J_cur = [j.cuda() for j in state["J_cur"]]
        print(f"resuming at layer {next_l}")
    else:
        w_norm = weights.tensor("model.norm.weight")
        J_sum = torch.zeros(n_layers, D, D)
        J_cur = []
        for st in stashes:
            h_final = st["h_final"].to("cuda", torch.float32)
            J = rms_norm_jacobian(h_final, w_norm, eps, valid_mask(h_final.shape[0]))
            J_cur.append(J)
            J_sum[n_layers - 1] += J.cpu()
        next_l = n_layers - 1

    for l in range(next_l, 0, -1):
        t0 = time.perf_counter()
        layer = build_true_layer(cfg, l, weights, torch.float32)
        for i, st in enumerate(stashes):
            h_in = st["h_in"][l].to("cuda", torch.float32)
            rope = (st["cos"].cuda(), st["sin"].cuda())
            M = decoder_layer_jacobian(layer, h_in, rope, valid_mask(h_in.shape[0]), chunk=chunk)
            J_cur[i] = J_cur[i] @ M
            J_sum[l - 1] += J_cur[i].cpu()
            del M
        del layer
        torch.cuda.empty_cache()
        print(f"layer {l}: {time.perf_counter() - t0:.0f}s "
              f"(||J_{l - 1}|| mean {J_sum[l - 1].norm() / len(stashes):.3e})", flush=True)
        if l % 4 == 0 or l == 1:
            torch.save({"next_l": l - 1 if l > 1 else 0, "J_sum": J_sum,
                        "J_cur": [j.cpu() for j in J_cur],
                        "corpus": [c.stem for c in corpus]}, CKPT)

    J_mean = J_sum / len(stashes)
    torch.save({
        "J": J_mean,  # (n_layers, D, D) fp32; J[l] transports layer-l OUTPUT
        "meta": {
            "lens": "j-lens-v1",
            "fit_date": date.today().isoformat(),
            "corpus": [c.stem for c in corpus],
            "n_prompts": len(corpus),
            "skip_first": 4,
            "caveats": [
                "fit on true bf16-checkpoint weights, applied to NF4-runtime "
                "activations (single-layer branch-delta drift 7-14%, reports/jlens_gate0.json)",
                "position/prompt-averaged Jacobians chained as prod of averages "
                "(branch junction ~1-4% rel per layer, layer 0 excluded from chain)",
                "corpus is synthetic 2-D clips + default question, prefill positions only",
            ],
            "attribution": "analytic recipe ported from WeZZard/jlens-qwen36 (Apache 2.0)",
        },
    }, out_path)
    print(f"wrote {out_path} ({J_mean.shape[0]} layers)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="jlens/j_lens_v1.pt")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--capture-only", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--clips", nargs="*", help="override corpus clip paths")
    args = ap.parse_args()

    corpus = [Path(c) for c in args.clips] if args.clips else default_corpus()
    print(f"corpus: {len(corpus)} clips")
    capture_corpus(corpus)
    if args.capture_only:
        return
    fit(corpus, Path(args.out), args.chunk, resume=not args.no_resume)


if __name__ == "__main__":
    main()
