"""J-lens unit tests (pure logic — no model, no GPU)."""

import torch
import pytest

from jsr.jacobian import rms_norm_jacobian, valid_mask
from jsr.labels import LAYER_FLOOR, extract_concepts
from jsr.lens import JLens


def test_valid_mask_skips_sinks_and_last():
    m = valid_mask(8, skip_first=4, device="cpu")
    assert m.tolist() == [0, 0, 0, 0, 1, 1, 1, 0]


def test_rms_norm_jacobian_matches_autograd_cpu():
    torch.manual_seed(0)
    x = torch.randn(6, 16, dtype=torch.float64)
    w = torch.rand(16, dtype=torch.float64) + 0.5
    eps = 1e-6

    def f(v):
        return v / torch.sqrt((v * v).mean() + eps) * w

    ref = torch.stack([torch.func.jacrev(f)(x[i]) for i in range(6)]).mean(0)
    analytic = rms_norm_jacobian(x.float(), w.float(), eps)
    assert ((analytic - ref.float()).norm() / ref.norm()).item() < 1e-5


def test_jlens_roundtrip_and_transport(tmp_path):
    torch.manual_seed(0)
    J = torch.randn(2, 4, 4)
    path = tmp_path / "j.pt"
    torch.save({"J": J, "meta": {"lens": "j-lens-v1", "caveats": ["c"]}}, path)
    lens = JLens.load(path, device="cpu")
    assert lens.meta["lens"] == "j-lens-v1"
    h = torch.randn(3, 4, dtype=torch.float16)
    got = lens.transport(h, 1)
    ref = h @ J[1].to(torch.float16).T
    assert got.dtype == torch.float16
    assert torch.allclose(got, ref, atol=1e-3)


def test_jlens_missing_file_error_names_fit_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="fit_jlens"):
        JLens.load(tmp_path / "nope.pt")


def _mini_trace(lens):
    return {
        "answer": "The ball falls.",
        "meta": {"lens": lens},
        "frame_groups": [{
            "group": 0,
            "raw_readouts": [
                {"layer": 10, "top_tokens": [" ball"], "strengths": [0.5]},
                {"layer": 26, "top_tokens": [" ball"], "strengths": [0.5]},
            ],
            "concepts": [],
        }],
    }


def test_layer_floor_keyed_by_trace_lens():
    baseline = {
        "layers": {layer: {"mean": 0.01, "std": 0.02} for layer in range(28)},
        "common_tokens": {},
    }
    trace = _mini_trace("logit-lens-v1")
    extract_concepts(trace, baseline=baseline)
    layers = {c["layer"] for c in trace["frame_groups"][0]["concepts"]}
    assert layers == {26}, "logit lens floor (20) must exclude the layer-10 readout"

    # a lens with a lower floor picks up the mid-layer readout too
    old = LAYER_FLOOR["j-lens-v1"]
    LAYER_FLOOR["j-lens-v1"] = 8
    try:
        trace = _mini_trace("j-lens-v1")
        extract_concepts(trace, baseline=baseline)
        layers = {c["layer"] for c in trace["frame_groups"][0]["concepts"]}
        assert 10 in layers, "j-lens floor (8) must include the layer-10 readout"
    finally:
        LAYER_FLOOR["j-lens-v1"] = old
