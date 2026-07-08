"""Label extraction unit tests (pure logic — no model)."""

from jsr.labels import canon, extract_concepts, phrase_candidates, wordlike, zscore


def make_baseline(mean=0.01, std=0.02, layers=range(28), common=None):
    return {
        "layers": {layer: {"mean": mean, "std": std} for layer in layers},
        "common_tokens": common or {},
    }


def make_trace(readouts_by_group):
    return {
        "answer": "The red ball rolls off the wet floor and falls.",
        "meta": {},
        "frame_groups": [
            {
                "group": gi,
                "raw_readouts": [
                    {"layer": layer, "top_tokens": toks, "strengths": shares}
                    for layer, toks, shares in group
                ],
                "concepts": [],
            }
            for gi, group in enumerate(readouts_by_group)
        ],
    }


def test_wordlike_and_canon():
    assert wordlike(" ball")
    assert not wordlike("<|object_ref_start|>")
    assert not wordlike("?>")
    assert not wordlike("换句话")
    assert canon(" Balls") == "ball"
    assert canon("rolling") == "roll"
    assert canon("red") == "red"


def test_phrase_candidates_include_answer_bigrams_and_general_list():
    cands = phrase_candidates({"answer": "The ball slipped on the wet floor."})
    assert "wet floor" in cands
    assert "ball" in cands
    assert "traffic light" in cands  # general list


def test_extract_concepts_matches_and_scores():
    trace = make_trace([
        [
            (26, [" red", " ball", " the", "junk123"], [0.09, 0.07, 0.30, 0.20]),
            (27, [" wet", " floor"], [0.05, 0.06]),
        ]
    ])
    extract_concepts(trace, baseline=make_baseline())
    concepts = trace["frame_groups"][0]["concepts"]
    labels = {c["label"] for c in concepts}
    assert "red ball" in labels  # bigram co-occurrence in the same cell
    assert "wet floor" in labels
    assert "the" not in labels  # stopword filtered
    rb = next(c for c in concepts if c["label"] == "red ball")
    assert rb["layer"] == 26
    assert set(rb["source_tokens"]) == {"red", "ball"}
    assert rb["strength"] == round((0.07 - 0.01) / 0.02, 2)  # min-word z


def test_unmatched_token_hidden_by_default_optin_surfaces_it():
    trace = make_trace([[(27, [" zebra"], [0.11])]])
    extract_concepts(trace, baseline=make_baseline())
    assert trace["frame_groups"][0]["concepts"] == []  # not in any candidate vocab
    extract_concepts(trace, baseline=make_baseline(), include_unmatched=True)
    labels = [c["label"] for c in trace["frame_groups"][0]["concepts"]]
    assert "zebra" in labels


def test_z_floor_suppresses_weak_readouts():
    trace = make_trace([[(27, [" ball"], [0.012])]])  # z = 0.1 < floor
    extract_concepts(trace, baseline=make_baseline())
    assert trace["frame_groups"][0]["concepts"] == []


def test_zscore_fallback_without_baseline():
    assert zscore(0.05, 3, {}) == 5.0


def test_common_token_suppressed_by_per_token_stats():
    # "registrazione" reads out everywhere in the baseline: its own mean is high,
    # so a typical share yields z ~ 0 while a content word scores high
    baseline = make_baseline(common={27: {"registrazione": {"mean": 0.09, "std": 0.02}}})
    trace = make_trace([[(27, [" registrazione", " ball"], [0.09, 0.09])]])
    extract_concepts(trace, baseline=baseline, include_unmatched=True)
    labels = {c["label"]: c["strength"] for c in trace["frame_groups"][0]["concepts"]}
    assert "registrazione" not in labels
    assert labels["ball"] == 4.0


def test_layer_floor_excludes_midlayer_junk():
    trace = make_trace([[(12, [" junkword"], [0.30]), (27, [" ball"], [0.09])]])
    extract_concepts(trace, baseline=make_baseline())
    labels = [c["label"] for c in trace["frame_groups"][0]["concepts"]]
    assert "junkword" not in labels
    assert "ball" in labels


def test_meta_updated_honestly():
    trace = make_trace([[(27, [" ball"], [0.09])]])
    extract_concepts(trace, baseline=make_baseline())
    assert "z-score" in trace["meta"]["strength_normalization"]
    assert trace["meta"]["label_extraction"] == "labels-v1"
