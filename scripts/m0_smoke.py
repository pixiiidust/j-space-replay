"""M0 smoke test: environment + model bring-up evidence (issue #2).

Runs, in order, resetting the CUDA peak-memory counter between stages:
  0. CUDA sanity: device name, capability (expect sm_120), driver-visible VRAM
  1. Load Qwen2.5-VL-7B-Instruct 4-bit (bitsandbytes NF4, attn_implementation="sdpa")
  2. Text-only forward pass (short generate)
  3. Single-image pass (synthetic red circle -> ask shape/color)
  4. 5 s video pass at 1 FPS / 480p / max_pixels cap -> record peak VRAM

Writes machine-readable evidence to reports/m0_results.json and prints a summary.

    uv run python scripts/m0_smoke.py [--clip fixtures/clips/ball_drop.mp4]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# ~250 visual tokens per 2-frame group: tokens = ceil(H/28)*ceil(W/28) after the
# 2x2 spatial merge, so cap pixels at ~250 * 28 * 28.
MAX_PIXELS = 250 * 28 * 28

results: dict = {"model": MODEL_ID, "stages": {}}


def gib(nbytes: int) -> float:
    return round(nbytes / 1024**3, 2)


def stage(name: str):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    def done(**extra):
        results["stages"][name] = {
            "seconds": round(time.perf_counter() - t0, 1),
            "peak_vram_allocated_gib": gib(torch.cuda.max_memory_allocated()),
            "peak_vram_reserved_gib": gib(torch.cuda.max_memory_reserved()),
            **extra,
        }
        print(f"[{name}] {results['stages'][name]}")

    return done


def default_5s_clip() -> Path:
    """The M0 task calls for a 5 s video pass; fixtures are 8-12 s, so cut a 5 s one."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from make_fixtures import _frames_ball_drop, write_clip

    path = Path("fixtures/clips/_m0_ball_drop_5s.mp4")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_clip(path, 5, _frames_ball_drop)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None)
    args = ap.parse_args()
    if args.clip is None:
        args.clip = str(default_5s_clip())

    assert torch.cuda.is_available(), "CUDA not available"
    cap = torch.cuda.get_device_capability()
    results["torch"] = torch.__version__
    results["cuda_device"] = torch.cuda.get_device_name(0)
    results["compute_capability"] = f"sm_{cap[0]}{cap[1]}"
    print(f"torch {torch.__version__} | {results['cuda_device']} | {results['compute_capability']}")
    assert cap == (12, 0), f"expected sm_120, got {cap}"

    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    done = stage("load_model")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        attn_implementation="sdpa",
        dtype=torch.float16,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
    attn_impl = getattr(model.config, "_attn_implementation", None) or getattr(
        model.config, "attn_implementation", "unknown"
    )
    done(attn_implementation=attn_impl, quant="bnb-nf4")

    def chat(messages, max_new_tokens=64):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        # transformers 5.x strictly validates processor kwargs: fps must be a scalar,
        # and qwen-vl-utils returns fps=[] when the prompt has no video.
        extra = {}
        if video_inputs:
            fps = video_kwargs.get("fps")
            if isinstance(fps, list):
                fps = fps[0] if fps else None
            if fps is not None:
                extra["fps"] = float(fps)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **extra,
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return (
            processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip(),
            inputs.input_ids.shape[1],
        )

    done = stage("text_pass")
    answer, n_tok = chat(
        [{"role": "user", "content": [{"type": "text", "text": "Reply with exactly one word: ready"}]}],
        max_new_tokens=8,
    )
    done(answer=answer, input_tokens=n_tok)

    done = stage("image_pass")
    img = Image.new("RGB", (448, 448), (255, 255, 255))
    ImageDraw.Draw(img).ellipse([124, 124, 324, 324], fill=(200, 30, 30))
    answer, n_tok = chat(
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "What shape is shown and what color is it? One short sentence."},
        ]}],
    )
    done(answer=answer, input_tokens=n_tok)

    done = stage("video_pass_5s")
    clip = Path(args.clip).resolve()
    assert clip.exists(), f"fixture clip missing: {clip} (run scripts/make_fixtures.py)"
    from jsr.video import sample_frames

    sampled = sample_frames(clip, fps=1.0)
    answer, n_tok = chat(
        [{"role": "user", "content": [
            {
                "type": "video",
                "video": sampled.frames,
                "max_pixels": MAX_PIXELS,
                "sample_fps": sampled.sample_fps,
            },
            {"type": "text", "text": "Describe what happens in this video."},
        ]}],
        max_new_tokens=96,
    )
    done(
        answer=answer,
        input_tokens=n_tok,
        clip=clip.name,
        sampled_frames=len(sampled.frames),
        sample_fps=round(sampled.sample_fps, 3),
        max_pixels=MAX_PIXELS,
    )

    peak = results["stages"]["video_pass_5s"]["peak_vram_reserved_gib"]
    results["acceptance"] = {
        "video_qa_end_to_end": True,
        "video_pass_peak_vram_gib": peak,
        "video_pass_peak_vram_le_12gib": peak <= 12.0,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/m0_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["acceptance"], indent=2))
    print("wrote reports/m0_results.json")


if __name__ == "__main__":
    main()
