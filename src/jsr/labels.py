"""M2 label extraction: raw token readouts -> derived concept labels.

Pipeline per SPEC §Label extraction:
  1. collect wordlike readout tokens per (frame group, layer) with patch-share
  2. filter stopwords/fragments; canonicalize (case, plural/inflection suffixes)
  3. z-score shares against a per-layer baseline corpus (fixtures/baseline_stats.json)
  4. build candidate phrase vocabulary: the model's own answer (+ optional
     caption pass when the model is available) + a small general concept list
  5. match single tokens and same-cell co-occurring token pairs against
     candidates; emit concepts with source_tokens provenance

Concept `strength` is a per-layer z-score of patch share — "readout strength",
never a probability. Raw readouts always stay alongside; the UI drills down.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "there", "with",
    "as", "by", "for", "from", "into", "over", "under", "then", "than", "but",
    "not", "no", "yes", "so", "if", "we", "you", "he", "she", "they", "what",
    "which", "who", "when", "where", "how", "why", "can", "could", "will",
    "would", "should", "may", "might", "must", "has", "have", "had", "do",
    "does", "did", "one", "two", "three", "same", "other", "also", "very",
    "more", "most", "some", "any", "all", "each", "both", "sees", "seems",
    "appears", "shows", "shown", "image", "video", "frame", "frames", "scene",
    "view", "part", "side", "long", "short", "small", "large", "simple",
}

# minimal general concept list (SPEC: "a general concept list") — extend freely
GENERAL_CONCEPTS = [
    "ball", "floor", "table", "wall", "door", "person", "hand", "car", "road",
    "traffic light", "red light", "green light", "square", "circle", "line",
    "background", "shadow", "edge", "corner", "falling", "rolling", "moving",
    "stopped", "bouncing", "growing", "shrinking", "appearing", "color change",
    "red", "green", "blue", "brown", "gray", "white", "black", "yellow",
]

# z-score baselines are LENS-SPECIFIC (shares change when the basis changes);
# make_baseline.py --lens ... regenerates the matching file
_BASELINE_PATHS = {
    "logit-lens-v1": Path(__file__).parent / "baseline_stats.json",
    "j-lens-v1": Path(__file__).parent / "baseline_stats_jlens.json",
}
_BASELINE_PATH = _BASELINE_PATHS["logit-lens-v1"]

# concept extraction band per lens: under the raw logit lens, content on
# visual tokens lives at layers ~20-27 (M1 risk gate). The J-lens moves the
# content onset down to ~22 (balance/ball/left readable at 22-23 on ball_drop,
# issue #8 evidence) but mid layers stay content-free (function words only),
# so its floor is 22 — not the hoped-for 8.
LAYER_FLOOR = {"logit-lens-v1": 20, "j-lens-v1": 22}


def wordlike(tok: str) -> bool:
    s = tok.strip()
    return len(s) >= 2 and s.isascii() and s.replace("-", "").isalpha()


def canon(tok: str) -> str:
    """Canonical word form: lowercase, strip trivial inflection suffixes."""
    s = tok.strip().lower()
    for suffix in ("ing", "ed", "es", "s"):
        if s.endswith(suffix) and len(s) - len(suffix) >= 3:
            s = s[: -len(suffix)]
            break
    return s


def load_baseline(path: str | Path | None = None, lens: str = "logit-lens-v1") -> dict:
    p = Path(path) if path else _BASELINE_PATHS.get(lens, _BASELINE_PATH)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("version") != 2:
        return {}
    return {
        "layers": {int(k): v for k, v in raw["layers"].items()},
        "common_tokens": {int(k): v for k, v in raw["common_tokens"].items()},
    }


def zscore(share: float, layer: int, baseline: dict, word: str | None = None) -> float:
    """z of a canonical word's patch share vs baseline. Words the baseline
    corpus reads out everywhere ("lens furniture": registrazione, wroc, ...)
    carry their own per-token stats, so they only score when a clip exceeds
    their usual share; content words absent from the baseline use the layer-
    wide stats and score high on genuinely elevated share."""
    if not baseline:
        return share / 0.01  # no baseline: crude scale so output stays usable
    tok_stats = baseline["common_tokens"].get(layer, {}).get(word) if word else None
    if tok_stats is not None:
        return (share - tok_stats["mean"]) / tok_stats["std"]
    stats = baseline["layers"].get(layer)
    if not stats or stats["std"] <= 0:
        return share / 0.01
    return (share - stats["mean"]) / stats["std"]


def phrase_candidates(trace: dict, caption: str | None = None) -> list[str]:
    """Candidate phrases from the model's own answer, an optional caption pass,
    and the general concept list. 1- and 2-grams of content words."""
    text = trace.get("answer", "") + " " + (caption or "")
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z-]+", text.lower()) if w not in STOPWORDS]
    grams: list[str] = []
    seen = set()
    for i, w in enumerate(words):
        for phrase in ([w] if len(w) >= 3 else []) + (
            [f"{w} {words[i + 1]}"] if i + 1 < len(words) else []
        ):
            if phrase not in seen:
                seen.add(phrase)
                grams.append(phrase)
    for c in GENERAL_CONCEPTS:
        if c not in seen:
            grams.append(c)
    return grams


def _cell_tokens(readout: dict) -> dict[str, tuple[float, str]]:
    """canonical word -> (max share, original token) for wordlike tokens of a cell."""
    out: dict[str, tuple[float, str]] = {}
    for tok, share in zip(readout["top_tokens"], readout["strengths"]):
        if not wordlike(tok):
            continue
        c = canon(tok)
        if c in STOPWORDS or len(c) < 2:
            continue
        if c not in out or share > out[c][0]:
            out[c] = (share, tok.strip())
    return out


def extract_concepts(
    trace: dict,
    baseline: dict | None = None,
    caption: str | None = None,
    z_floor: float = 0.6,
    max_per_group: int = 12,
    layer_floor: int | None = None,
    include_unmatched: bool = False,
) -> None:
    """Fill trace[...]["concepts"] in place from raw readouts.

    Matching: a 1-gram candidate matches a canonical token; a 2-gram matches
    when both words appear in the SAME (group, layer) cell (co-occurrence).
    Only candidate-matched readouts become concepts by default (SPEC §Label
    extraction): the candidate vocabulary (answer + caption + curated list) is
    what separates content from lens junk that survives z-scoring (quality-gate
    finding). Set include_unmatched=True to also surface unmatched high-z
    tokens as their own labels — noisy, but nothing is hidden either way: the
    full raw readouts stay in the trace for drill-down.

    layer_floor: concepts come from the late-layer band only; defaults to the
    LAYER_FLOOR entry for the trace's lens (M1 risk gate: content readouts on
    visual tokens live at layers ~20-27 under the raw logit lens; the J-lens
    band comes from its own #8 evidence). Raw readouts for ALL layers stay in
    the trace regardless.
    """
    lens = trace.get("meta", {}).get("lens", "logit-lens-v1")
    if layer_floor is None:
        layer_floor = LAYER_FLOOR.get(lens, 20)
    baseline = baseline if baseline is not None else load_baseline(lens=lens)
    candidates = phrase_candidates(trace, caption)
    cand_words = {c: c.split() for c in candidates}

    for g in trace["frame_groups"]:
        scored: dict[str, dict] = {}
        for readout in g["raw_readouts"]:
            layer = readout["layer"]
            if layer < layer_floor:
                continue
            cell = _cell_tokens(readout)
            if not cell:
                continue
            zs = {w: zscore(share, layer, baseline, word=w) for w, (share, _) in cell.items()}
            matched_words: set[str] = set()
            for cand, words in cand_words.items():
                cwords = [canon(w) for w in words]
                if all(w in cell for w in cwords):
                    z = min(zs[w] for w in cwords)  # phrase is as strong as its weakest word
                    if z < z_floor:
                        continue
                    matched_words.update(cwords)
                    prev = scored.get(cand)
                    if prev is None or z > prev["strength"]:
                        scored[cand] = {
                            "label": cand,
                            "strength": round(z, 2),
                            "layer": layer,
                            "source_tokens": [cell[w][1] for w in cwords],
                        }
            if not include_unmatched:
                continue
            for w, z in zs.items():  # opt-in: unmatched high-z tokens as themselves
                if w in matched_words or z < z_floor or len(w) < 3:
                    continue
                prev = scored.get(w)
                if prev is None or z > prev["strength"]:
                    scored[w] = {
                        "label": w,
                        "strength": round(z, 2),
                        "layer": layer,
                        "source_tokens": [cell[w][1]],
                    }
        g["concepts"] = sorted(scored.values(), key=lambda c: -c["strength"])[:max_per_group]

    trace["meta"]["strength_normalization"] = (
        "concepts: per-layer z-score of patch-share vs baseline corpus; "
        "raw visual readouts: patch-share; answer readouts: raw logit"
    )
    trace["meta"]["label_extraction"] = "labels-v1"
    # M2 quality-gate outcome (reports/m2_quality_gate.md): shown concepts are
    # ~92% sensible but boards are sparse (1-5 per clip) — the "6 of top-10"
    # gate fails on recall. Per issue #4 fallback: concepts are EXPERIMENTAL,
    # the raw-token grid remains the primary view.
    trace["meta"]["concepts_quality"] = "experimental (high precision, low recall - see M2 gate)"


def caption_pass(clip, model, processor, max_new_tokens: int = 96) -> str:
    """One cheap extra generate: ask the model to enumerate visible content."""
    import torch
    from qwen_vl_utils import process_vision_info

    from jsr.model import MAX_PIXELS
    from jsr.video import sample_frames

    sampled = sample_frames(clip, fps=1.0)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": sampled.frames, "max_pixels": MAX_PIXELS,
         "sample_fps": sampled.sample_fps},
        {"type": "text", "text": "List the objects, colors, and events visible in this video "
                                 "as short comma-separated phrases. No sentences."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, videos, vk = process_vision_info(messages, return_video_kwargs=True)
    fps_kw = vk.get("fps")
    if isinstance(fps_kw, list):
        fps_kw = fps_kw[0] if fps_kw else None
    inputs = processor(
        text=[text], videos=videos, return_tensors="pt",
        **({"fps": float(fps_kw)} if fps_kw is not None else {}),
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def add_concepts(trace: dict, model=None, processor=None, clip=None, baseline_path=None) -> dict:
    """Entry point used by the pipeline/server. Caption pass only when the
    model and clip are available; degrades gracefully without them."""
    caption = None
    if model is not None and processor is not None and clip is not None:
        caption = caption_pass(clip, model, processor)
        trace["meta"]["caption"] = caption
    lens = trace.get("meta", {}).get("lens", "logit-lens-v1")
    extract_concepts(trace, baseline=load_baseline(baseline_path, lens=lens), caption=caption)
    return trace
