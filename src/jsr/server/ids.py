"""Pinned id derivations — the API contract the M4 frontend is built against.

- video_id  = sha256(file bytes)[:16]           (first 16 hex chars)
- q_hash    = sha256(question utf-8)[:8]         (first 8 hex chars)
- trace_id  = f"{video_id}-{q_hash}"

Kept in one tiny module so every caller (upload, trace request, cache lookup,
library) derives ids identically.
"""

from __future__ import annotations

import hashlib

DEFAULT_QUESTION = "Describe what happens in this video."


def video_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]


def trace_id_for(video_id: str, question: str) -> str:
    return f"{video_id}-{question_hash(question)}"
