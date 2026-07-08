"""Golden tests: full pipeline on the 3 fixture clips vs committed expected shapes.

GPU-only (marked `gpu`): runs the real model. Structural expectations only —
token counts, grid dims, timing map — never generated text.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="golden trace tests need the GPU + model"
)

GOLDEN = sorted(Path("fixtures/golden").glob("*.golden.json"))


@pytest.fixture(scope="session")
def model_and_processor():
    from jsr.model import load_model_and_processor

    return load_model_and_processor()


@pytest.mark.gpu
@pytest.mark.parametrize("golden_path", GOLDEN, ids=lambda p: p.stem)
def test_trace_matches_golden_shape(golden_path, model_and_processor):
    from jsr.schema import trace_shape, validate_trace
    from jsr.trace import run_trace

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    model, processor = model_and_processor
    clip = Path("fixtures/clips") / f"{golden['video_id']}.mp4"
    trace = run_trace(clip, golden["question"], model=model, processor=processor)
    validate_trace(trace)

    assert trace_shape(trace) == golden


@pytest.mark.gpu
def test_answer_tokens_align_with_readouts(model_and_processor):
    """Every generated token has per-layer readouts; late layers include the token itself
    somewhere in the trace (plumbing check, not a semantic guarantee)."""
    from jsr.trace import run_trace

    model, processor = model_and_processor
    trace = run_trace(
        "fixtures/clips/ball_drop.mp4",
        "What color is the ball? Answer in one word.",
        model=model,
        processor=processor,
        max_new_tokens=8,
    )
    assert trace["answer_tokens"], "no answer-token readouts captured"
    n_layers = trace["meta"]["n_layers"]
    for at in trace["answer_tokens"]:
        assert len(at["readouts_by_layer"]) == n_layers
    # the final layer's top-1 readout should be the emitted token for most steps
    hits = sum(
        1
        for at in trace["answer_tokens"]
        if at["readouts_by_layer"][str(n_layers - 1)]["top_tokens"][0].strip()
        == at["token"].strip()
    )
    assert hits / len(trace["answer_tokens"]) >= 0.5
