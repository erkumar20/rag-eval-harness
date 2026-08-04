from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from eval_harness.dataset.schema import QAPair, QuestionCategory
from eval_harness.interfaces.pipeline import RAGPipelineInterface
from eval_harness.judge.base import JudgeLLM
from eval_harness.judge.schema import Diagnosis
from eval_harness.metrics.base import Metric
from eval_harness.schemas import EvalInput, EvalResult, MetricResult


class FakePipeline:
    """No inheritance from RAGPipelineInterface -- proves the contract is structural
    (duck-typed), which is what lets a thin adapter plug any pipeline in without
    coupling it to this package's class hierarchy."""

    def query(self, question: str) -> dict:
        return {"answer": f"answer to: {question}", "contexts": ["chunk-1", "chunk-2"]}


class NotAPipeline:
    def run(self, question: str) -> dict:
        return {}


def test_pipeline_interface_is_structural():
    assert isinstance(FakePipeline(), RAGPipelineInterface)
    assert not isinstance(NotAPipeline(), RAGPipelineInterface)


def test_pipeline_interface_output_shape():
    result = FakePipeline().query("what is RAG?")
    assert set(result.keys()) == {"answer", "contexts"}
    assert isinstance(result["contexts"], list)


def test_qa_pair_valid():
    qa = QAPair(
        id="q001",
        question="What is faithfulness?",
        ground_truth_answer="A RAGAS metric measuring groundedness in retrieved context.",
        ground_truth_contexts=["chunk about faithfulness"],
        category=QuestionCategory.DIRECT,
        difficulty="easy",
    )
    assert qa.category == QuestionCategory.DIRECT
    assert qa.difficulty == "easy"


def test_qa_pair_requires_question():
    with pytest.raises(ValidationError):
        QAPair(id="q002", ground_truth_answer="missing the question field")  # type: ignore[call-arg]


class FakeMetric(Metric):
    name = "fake_metric"

    def score(self, eval_input: EvalInput) -> MetricResult:
        return MetricResult(metric_name=self.name, score=1.0 if eval_input.answer else 0.0)


def test_metric_abc_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Metric()  # type: ignore[abstract]


def test_metric_subclass_works():
    metric = FakeMetric()
    result = metric.score(EvalInput(question="q", answer="a", contexts=["c"]))
    assert result.score == 1.0
    assert result.metric_name == "fake_metric"


class FakeJudge(JudgeLLM):
    def judge(
        self, question: str, contexts: list[str], answer: str, metric_name: str, score: float
    ) -> Diagnosis:
        return Diagnosis(
            metric=metric_name,
            verdict="fail" if score < 0.5 else "pass",
            explanation="stub diagnosis for testing",
        )


def test_judge_abc_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        JudgeLLM()  # type: ignore[abstract]


def test_judge_subclass_works():
    judge = FakeJudge()
    diagnosis = judge.judge("q", ["c"], "a", "faithfulness", 0.2)
    assert diagnosis.verdict == "fail"
    assert diagnosis.metric == "faithfulness"


def test_eval_result_defaults_are_independent_and_populated():
    r1 = EvalResult(qa_id="q001", question="q", answer="a", contexts=["c"], pipeline_version="v1")
    r2 = EvalResult(qa_id="q002", question="q2", answer="a2", contexts=["c2"], pipeline_version="v1")

    r1.metric_results.append(MetricResult(metric_name="faithfulness", score=0.9))

    # Would fail if Pydantic's default_factory were secretly a shared mutable default.
    assert r2.metric_results == []
    assert r2.judge_diagnoses == []
    assert isinstance(r1.timestamp, datetime)
