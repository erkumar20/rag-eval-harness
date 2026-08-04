from __future__ import annotations

from abc import ABC, abstractmethod

from eval_harness.judge.schema import Diagnosis


class JudgeLLM(ABC):
    """Contract for any LLM-as-judge implementation (GPT-4o now, others later)."""

    @abstractmethod
    def judge(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        metric_name: str,
        score: float,
    ) -> Diagnosis: ...
