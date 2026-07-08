"""Trace schema validation (the contract M3/M4 build against)."""

import pytest

from jsr.schema import SCHEMA_VERSION, validate_trace


def minimal_trace():
    return {
        "schema": SCHEMA_VERSION,
        "video_id": "clip",
        "question": "q",
        "answer": "a",
        "meta": {
            "model": "qwen2.5-vl-7b-nf4",
            "lens": "logit-lens-v1",
            "temporal_resolution_frames": 2,
            "strength_normalization": "raw-logit",
            "n_layers": 28,
        },
        "frame_groups": [
            {
                "group": 0,
                "time_start": 0.0,
                "time_end": 2.0,
                "frame_indices": [0, 1],
                "raw_readouts": [
                    {"layer": 0, "top_tokens": ["a"], "strengths": [1.0]},
                ],
                "concepts": [],
            }
        ],
        "answer_tokens": [
            {"token": "a", "readouts_by_layer": {"0": {"top_tokens": ["a"], "strengths": [1.0]}}}
        ],
        "grounding": [],
    }


def test_valid_trace_passes():
    validate_trace(minimal_trace())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.pop("schema"),
        lambda t: t.pop("frame_groups"),
        lambda t: t["frame_groups"][0].pop("raw_readouts"),
        lambda t: t["frame_groups"][0]["raw_readouts"][0].update(
            top_tokens=["a", "b"], strengths=[1.0]
        ),  # length mismatch
        lambda t: t["meta"].pop("lens"),
        lambda t: t.update(schema=99),
    ],
)
def test_invalid_traces_fail(mutate):
    t = minimal_trace()
    mutate(t)
    with pytest.raises(ValueError):
        validate_trace(t)
