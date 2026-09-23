"""Integration tests for the Phase 9 runner: fakes for the pipeline/metrics/judge (no live
LLM calls, matching the pattern in tests/unit/test_interfaces.py), but real Postgres writes
via the actual Phase 8 storage layer -- the runner's job is exactly to wire those two
together, so faking storage too would leave that wiring unverified."""

from __future__ import annotations

from sqlalchemy import select

from eval_harness.dataset.schema import QAPair, QuestionCategory
from eval_harness.judge.base import JudgeLLM
from eval_harness.judge.schema import Diagnosis
from eval_harness.metrics.base import Metric
from eval_harness.runner.baseline import check_baseline_gate, set_baseline_from_run
from eval_harness.runner.evaluate import run_evaluation
from eval_harness.schemas import EvalInput, MetricResult
from eval_harness.storage.db import session_scope
from eval_harness.storage.models import Result


class FakePipeline:
    def query(self, question: str) -> dict:
        # Deliberately answers "badly" for one known question so tests can assert the
        # judge only fires on that one and the mean score reflects the mix.
        if "bad" in question:
            return {"answer": "a fabricated answer", "contexts": ["unrelated context"]}
        return {"answer": f"good answer to: {question}", "contexts": ["relevant context"]}


class ScoreByAnswerMetric(Metric):
    """Fake metric: scores 0.2 for the known-bad answer, 1.0 otherwise -- lets tests assert
    on judge-triggering and mean-score computation without any real RAGAS/LLM call."""

    name = "fake_score"

    def score(self, eval_input: EvalInput) -> MetricResult:
        value = 0.2 if "fabricated" in eval_input.answer else 1.0
        return MetricResult(metric_name=self.name, score=value)


class AlwaysFailsMetric(Metric):
    """A metric that never applies to this pipeline (e.g. HallucinationMetric on a
    non-structured pipeline) -- proves the runner skips it instead of crashing the run."""

    name = "inapplicable"

    def score(self, eval_input: EvalInput) -> MetricResult:
        raise ValueError("this metric requires structured output this pipeline never produces")


class RecordingFakeJudge(JudgeLLM):
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def judge(
        self, question: str, contexts: list[str], answer: str, metric_name: str, score: float
    ) -> Diagnosis:
        self.calls.append((metric_name, score))
        return Diagnosis(metric=metric_name, verdict="fail", explanation="fabricated content")


def _dataset() -> list[QAPair]:
    return [
        QAPair(
            id="r001",
            question="a good question",
            ground_truth_answer="doesn't matter for this fake metric",
            category=QuestionCategory.DIRECT,
        ),
        QAPair(
            id="r002",
            question="a bad question",
            ground_truth_answer="doesn't matter for this fake metric",
            category=QuestionCategory.ADVERSARIAL,
        ),
    ]


def test_run_evaluation_scores_persists_and_invokes_judge_only_on_failures(
    pipeline_name: str,
) -> None:
    judge = RecordingFakeJudge()
    summary = run_evaluation(
        FakePipeline(),
        _dataset(),
        [ScoreByAnswerMetric(), AlwaysFailsMetric()],
        judge=judge,
        pipeline_name=pipeline_name,
        pipeline_version="v1",
        dataset_version="test-v1",
    )

    assert summary.mean_scores == {"fake_score": 0.6}
    assert judge.calls == [("fake_score", 0.2)]

    with session_scope() as session:
        results = session.scalars(select(Result).where(Result.run_id == summary.run_id)).all()
        assert len(results) == 2
        by_id = {r.question_id: r for r in results}
        assert by_id["r001"].metric_scores == {"fake_score": 1.0}
        assert by_id["r002"].metric_scores == {"fake_score": 0.2}
        assert by_id["r002"].judge_diagnosis[0]["verdict"] == "fail"
        assert by_id["r001"].judge_diagnosis is None


def test_baseline_gate_passes_with_no_prior_baseline_then_catches_a_real_regression(
    pipeline_name: str,
) -> None:
    good_pipeline_summary = run_evaluation(
        FakePipeline(),
        _dataset()[:1],
        [ScoreByAnswerMetric()],
        pipeline_name=pipeline_name,
        pipeline_version="v1",
        dataset_version="test-v1",
    )
    first_gate = check_baseline_gate(good_pipeline_summary)
    assert first_gate.passed is True
    assert first_gate.metrics[0].baseline_value is None

    set_baseline_from_run(good_pipeline_summary)

    class RegressedPipeline:
        def query(self, question: str) -> dict:
            return {"answer": "a fabricated answer", "contexts": ["unrelated"]}

    regressed_summary = run_evaluation(
        RegressedPipeline(),
        _dataset()[:1],
        [ScoreByAnswerMetric()],
        pipeline_name=pipeline_name,
        pipeline_version="v2-regressed",
        dataset_version="test-v1",
    )
    second_gate = check_baseline_gate(regressed_summary)
    assert second_gate.passed is False
    assert second_gate.metrics[0].baseline_value == 1.0
    assert second_gate.metrics[0].percent_drop == 0.8
