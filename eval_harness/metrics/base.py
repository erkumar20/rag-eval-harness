from __future__ import annotations

from abc import ABC, abstractmethod

from eval_harness.schemas import EvalInput, MetricResult


class Metric(ABC):
    """Contract every metric (RAGAS-backed or custom) must implement."""

    name: str

    @abstractmethod
    def score(self, eval_input: EvalInput) -> MetricResult: ...
