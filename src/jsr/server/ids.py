"""Pinned id derivations — the API contract the M4 frontend is built against.

- video_id  = sha256(file bytes)[:16]           (first 16 hex chars)
- q_hash    = sha256(question utf-8)[:8]         (first 8 hex chars)
- trace_id  = f"{video_id}-{q_hash}"             (default logit lens)
- trace_id  = f"{video_id}-{q_hash}-jl"          (lens "j-lens-v1")

The lens suffix keeps a J-lens trace of the same (video, question) from
colliding with — and silently replacing — the logit-lens trace in the cache.

Kept in one tiny module so every caller (upload, trace request, cache lookup,
library) derives ids identically.
"""

from __future__ import annotations

import hashlib

DEFAULT_QUESTION = "Describe what happens in this video."
DEFAULT_LENS = "logit-lens-v1"
LENSES = ("logit-lens-v1", "j-lens-v1")
_LENS_SUFFIX = {"logit-lens-v1": "", "j-lens-v1": "-jl"}


def video_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]


def trace_id_for(video_id: str, question: str, lens: str = DEFAULT_LENS) -> str:
    return f"{video_id}-{question_hash(question)}{_LENS_SUFFIX[lens]}"
