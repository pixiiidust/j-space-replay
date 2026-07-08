"""Model-native grounding queries for the top ~5 concepts.

Qwen2.5-VL can be asked directly for bounding boxes ("model-native grounding"),
which is separate from the lens readouts (see SPEC: "Clean bounding boxes come
from Qwen2.5-VL's native grounding ... as a separate overlay channel").

`run_grounding` picks the strongest concept labels from the trace and, for each,
asks the model for a box via `query_fn`. `query_fn` is the ONE function that
touches the model — tests inject a fake so nothing here loads CUDA. The default
`make_query_fn` builds the real closure that samples the concept's peak frame
and runs a grounding-prompt generation.

Grounding entries are shaped exactly:
    {"label", "box": [x1, y1, x2, y2], "time": float, "source": "model-grounding-query"}
"""

from __future__ import annotations

import json
import re
from typing import Callable

GROUNDING_SOURCE = "model-grounding-query"

# query_fn signature: (label: str, time: float) -> str   (raw model response text)
QueryFn = Callable[[str, float], str]


def grounding_prompt(label: str) -> str:
    """Qwen2.5-VL native-grounding prompt asking for a JSON bbox."""
    return (
        f"Outline the position of {label} in the image and output its bounding "
        "box coordinates in JSON format as "
        '[{"bbox_2d": [x1, y1, x2, y2], "label": "'
        f"{label}"
        '"}].'
    )


def parse_box(text: str) -> list[float] | None:
    """Pull the first [x1,y1,x2,y2] box out of a Qwen grounding response.

    Handles the documented `[{"bbox_2d": [...], "label": ...}]` shape as well as
    a bare 4-number list, and tolerates markdown ```json fences.
    """
    if not text:
        return None
    cleaned = text.strip().strip("`")
    cleaned = re.sub(r"^json\s*", "", cleaned, flags=re.IGNORECASE)
    box = _box_from_json(cleaned)
    if box is None:
        box = _box_from_regex(text)
    if box is None or len(box) != 4:
        return None
    return [float(v) for v in box]


def _box_from_json(text: str):
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if isinstance(data, dict):
        for key in ("bbox_2d", "bbox", "box"):
            if isinstance(data.get(key), list) and len(data[key]) == 4:
                return data[key]
        return None
    if isinstance(data, list) and len(data) == 4 and all(isinstance(v, (int, float)) for v in data):
        return data
    return None


def _box_from_regex(text: str):
    m = re.search(r"\[\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){3})\s*\]", text)
    if not m:
        return None
    return [float(v) for v in m.group(1).split(",")]


def top_concepts(trace: dict, top_k: int = 5) -> list[tuple[str, float]]:
    """Strongest unique concept labels across all frame groups, with peak time.

    Returns [(label, time), ...]. Empty when M2 label extraction hasn't run
    (concepts are absent), in which case grounding is simply a no-op.
    """
    best: dict[str, tuple[float, float]] = {}  # label -> (strength, time)
    for g in trace.get("frame_groups", []):
        t = (g.get("time_start", 0.0) + g.get("time_end", 0.0)) / 2.0
        for c in g.get("concepts", []) or []:
            label = c.get("label")
            if not label:
                continue
            strength = float(c.get("strength", 0.0))
            if label not in best or strength > best[label][0]:
                best[label] = (strength, t)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    return [(label, time) for label, (_strength, time) in ranked[:top_k]]


def run_grounding(trace: dict, *, query_fn: QueryFn, top_k: int = 5) -> dict:
    """Append model-native grounding boxes for the top ~`top_k` concepts."""
    grounding = trace.setdefault("grounding", [])
    for label, time in top_concepts(trace, top_k):
        try:
            raw = query_fn(label, time)
        except Exception:  # noqa: BLE001 - one bad box must not sink the trace
            continue
        box = parse_box(raw)
        if box is None:
            continue
        grounding.append(
            {"label": label, "box": box, "time": round(float(time), 3), "source": GROUNDING_SOURCE}
        )
    return trace


def make_query_fn(model, processor, video_path, *, max_new_tokens: int = 128) -> QueryFn:
    """Real grounding closure: sample the peak frame and run one generation.

    Isolated here so `run_grounding` never imports torch/qwen; only the returned
    closure (which tests replace) does the model work.
    """

    def query_fn(label: str, time: float) -> str:
        import torch

        from jsr.video import sample_frames

        sampled = sample_frames(video_path, fps=1.0)
        if not sampled.frames:
            return ""
        # Frame nearest the concept's peak time.
        idx = min(
            range(len(sampled.timestamps)),
            key=lambda i: abs(sampled.timestamps[i] - time),
        )
        image = sampled.frames[idx]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": grounding_prompt(label)},
                ],
            }
        ]
        from qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1] :]
        return processor.tokenizer.decode(gen, skip_special_tokens=True)

    return query_fn
