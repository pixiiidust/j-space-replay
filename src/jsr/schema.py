"""Trace schema (version 1) and validation — the contract the backend/frontend build on.

Kept dependency-free (no jsonschema) so the check runs everywhere, including
inside the trace writer itself before anything hits disk.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

_META_REQUIRED = {"model", "lens", "temporal_resolution_frames", "strength_normalization", "n_layers"}
_TOP_REQUIRED = {"schema", "video_id", "question", "answer", "meta", "frame_groups", "answer_tokens", "grounding"}
_GROUP_REQUIRED = {"group", "time_start", "time_end", "frame_indices", "raw_readouts", "concepts"}


def _fail(msg: str):
    raise ValueError(f"invalid trace: {msg}")


def _check_readout(r: dict, where: str):
    if not isinstance(r.get("top_tokens"), list) or not isinstance(r.get("strengths"), list):
        _fail(f"{where}: readout needs top_tokens + strengths lists")
    if len(r["top_tokens"]) != len(r["strengths"]):
        _fail(f"{where}: top_tokens/strengths length mismatch")


def trace_shape(trace: dict) -> dict:
    """Structural fingerprint of a trace — what the golden tests lock down."""
    return {
        "video_id": trace["video_id"],
        "question": trace["question"],
        "n_layers": trace["meta"]["n_layers"],
        "sampled_frames": trace["meta"]["sampled_frames"],
        "video_grid_thw": trace["meta"]["video_grid_thw"],
        "visual_tokens": trace["meta"]["visual_tokens"],
        "input_tokens": trace["meta"]["input_tokens"],
        "n_groups": len(trace["frame_groups"]),
        "patch_grid": trace["frame_groups"][0]["patch_grid"],
        "tokens_per_group": trace["frame_groups"][0]["n_patch_tokens"],
        "group_times": [[g["time_start"], g["time_end"]] for g in trace["frame_groups"]],
    }


def validate_trace(trace: dict) -> None:
    missing = _TOP_REQUIRED - trace.keys()
    if missing:
        _fail(f"missing top-level keys {sorted(missing)}")
    if trace["schema"] != SCHEMA_VERSION:
        _fail(f"schema {trace['schema']!r} != supported {SCHEMA_VERSION}")
    meta_missing = _META_REQUIRED - trace["meta"].keys()
    if meta_missing:
        _fail(f"meta missing {sorted(meta_missing)}")
    if not isinstance(trace["frame_groups"], list) or not trace["frame_groups"]:
        _fail("frame_groups must be a non-empty list")
    for g in trace["frame_groups"]:
        gm = _GROUP_REQUIRED - g.keys()
        if gm:
            _fail(f"frame group missing {sorted(gm)}")
        if not isinstance(g["raw_readouts"], list):
            _fail("raw_readouts must be a list")
        for r in g["raw_readouts"]:
            if "layer" not in r:
                _fail("raw readout missing layer")
            _check_readout(r, f"group {g['group']} layer {r.get('layer')}")
    for at in trace["answer_tokens"]:
        if "token" not in at or "readouts_by_layer" not in at:
            _fail("answer token entries need token + readouts_by_layer")
        for layer, r in at["readouts_by_layer"].items():
            _check_readout(r, f"answer token {at['token']!r} layer {layer}")
