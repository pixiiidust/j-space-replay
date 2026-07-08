"""M3 backend service: FastAPI upload/ingest, single-GPU job queue, trace store.

Nothing here loads the model or touches CUDA at import time — the pipeline
imports `jsr.trace.run_trace` and calls `load_model_and_processor` lazily, on
the first job only (see `pipeline.Pipeline`).
"""
