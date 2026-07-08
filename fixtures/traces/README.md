# Fixture traces

Real pipeline output (`python -m jsr.trace`, default question) on the three
synthetic fixture clips — committed so the backend (M3) and frontend (M4) can
be developed and tested **without a GPU**. Schema v1 (`src/jsr/schema.py`).

Regenerate after pipeline changes:

    uv run python scripts/make_fixtures.py
    uv run python scripts/make_golden.py   # also refreshes reports/trace_*_default.json
    # then copy reports/trace_<name>_default.json -> fixtures/traces/<name>.trace.json

`concepts` and `grounding` are empty until M2/M3 fill them; treat presence of
those keys, not their contents, as the contract.
