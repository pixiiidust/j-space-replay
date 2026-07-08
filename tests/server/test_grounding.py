"""Grounding: box parsing, top-concept selection, and the mocked query loop.

The model call goes through `query_fn`; tests inject a fake, so nothing here
loads the model or touches CUDA.
"""

from __future__ import annotations

from jsr.server.grounding import (
    GROUNDING_SOURCE,
    grounding_prompt,
    parse_box,
    run_grounding,
    top_concepts,
)


def _trace_with_concepts():
    return {
        "frame_groups": [
            {
                "time_start": 0.0,
                "time_end": 2.0,
                "concepts": [
                    {"label": "red ball", "strength": 2.9},
                    {"label": "table", "strength": 1.1},
                ],
            },
            {
                "time_start": 2.0,
                "time_end": 4.0,
                "concepts": [
                    {"label": "red ball", "strength": 3.4},  # stronger peak later
                    {"label": "shadow", "strength": 2.0},
                ],
            },
        ],
        "grounding": [],
    }


def test_parse_box_qwen_bbox_2d():
    text = '[{"bbox_2d": [12, 30, 480, 420], "label": "floor"}]'
    assert parse_box(text) == [12.0, 30.0, 480.0, 420.0]


def test_parse_box_with_markdown_fence():
    text = '```json\n[{"bbox_2d": [1, 2, 3, 4], "label": "x"}]\n```'
    assert parse_box(text) == [1.0, 2.0, 3.0, 4.0]


def test_parse_box_bare_list():
    assert parse_box("[5, 6, 7, 8]") == [5.0, 6.0, 7.0, 8.0]


def test_parse_box_freeform_prose():
    assert parse_box("The box is [10, 20, 30, 40] roughly.") == [10.0, 20.0, 30.0, 40.0]


def test_parse_box_garbage_returns_none():
    assert parse_box("no coordinates here") is None
    assert parse_box("") is None


def test_top_concepts_ranks_by_peak_strength_and_dedupes():
    got = top_concepts(_trace_with_concepts(), top_k=5)
    labels = [label for label, _t in got]
    assert labels[0] == "red ball"  # 3.4 peak beats everything
    assert labels.count("red ball") == 1  # deduped across groups
    # red ball's peak is in the 2nd group (t = 3.0)
    assert dict(got)["red ball"] == 3.0


def test_top_concepts_empty_when_no_concepts():
    assert top_concepts({"frame_groups": [{"concepts": []}]}) == []


def test_run_grounding_appends_shaped_entries():
    trace = _trace_with_concepts()
    seen = []

    def fake_query(label, time):
        seen.append((label, time))
        return f'[{{"bbox_2d": [0, 0, 100, 100], "label": "{label}"}}]'

    run_grounding(trace, query_fn=fake_query, top_k=3)

    assert len(trace["grounding"]) == 3
    entry = trace["grounding"][0]
    assert set(entry) == {"label", "box", "time", "source"}
    assert entry["source"] == GROUNDING_SOURCE
    assert entry["box"] == [0.0, 0.0, 100.0, 100.0]
    assert entry["label"] == "red ball"


def test_run_grounding_skips_unparseable_and_survives_errors():
    trace = _trace_with_concepts()

    def flaky_query(label, time):
        if label == "red ball":
            raise RuntimeError("model hiccup")
        if label == "shadow":
            return "no box"
        return "[1, 2, 3, 4]"

    run_grounding(trace, query_fn=flaky_query, top_k=5)
    labels = [e["label"] for e in trace["grounding"]]
    assert "red ball" not in labels  # errored out, skipped
    assert "shadow" not in labels  # unparseable, skipped
    assert "table" in labels


def test_grounding_prompt_mentions_label_and_json():
    p = grounding_prompt("wet floor")
    assert "wet floor" in p
    assert "JSON" in p or "json" in p
