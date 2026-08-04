from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from eval_harness.judge.schema import Diagnosis


class EvalInput(BaseModel):
    """Standardized input a Metric scores: a pipeline's actual output for one question,
    plus the ground truth from the QAPair it was run against."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth_answer: str | None = None
    ground_truth_contexts: list[str] = Field(default_factory=list)


class MetricResult(BaseModel):
    metric_name: str
    score: float
    raw: dict | None = None


class EvalResult(BaseModel):
    """Full record of evaluating one golden-dataset question against one pipeline run."""

    qa_id: str
    question: str
    answer: str
    contexts: list[str]
    pipeline_version: str
    dataset_version: str | None = None
    metric_results: list[MetricResult] = Field(default_factory=list)
    judge_diagnoses: list[Diagnosis] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
