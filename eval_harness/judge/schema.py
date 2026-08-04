from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Diagnosis(BaseModel):
    """Structured explanation of why a metric scored low on a given question, produced by
    the judge LLM. Pulled forward from Phase 6 because JudgeLLM (Phase 2) needs a concrete
    return type to be a real interface rather than a stub.
    """

    metric: str
    verdict: Literal["pass", "fail"]
    explanation: str
    evidence_span: str | None = None
