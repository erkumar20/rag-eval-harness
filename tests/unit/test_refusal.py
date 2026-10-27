"""Tests for how unanswerable questions are scored.

A correct refusal is the right behaviour on an out-of-scope question, but RAGAS's
answer_relevance reads it as "evasive" and forces the score to 0. So unanswerable questions are
scored by a dedicated refusal check, and the metrics that cannot handle refusals skip them.
"""

from __future__ import annotations

import pytest

from eval_harness.config import Settings
from eval_harness.metrics.ragas_metrics import AnswerRelevanceMetric
from eval_harness.metrics.refusal import RefusalCorrectnessMetric, RefusalVerdict
from eval_harness.runner.evaluate import mean_scores_by_category
from eval_harness.schemas import EvalInput


def _classifier(declines: bool):
    return lambda question, answer: RefusalVerdict(declines=declines, reason="stub")


def _unanswerable(answer: str) -> EvalInput:
    return EvalInput(question="How do I do X?", answer=answer, contexts=[], expects_refusal=True)


def test_a_correct_refusal_scores_one() -> None:
    metric = RefusalCorrectnessMetric(classifier=_classifier(True))
    result = metric.score(_unanswerable("The excerpts do not cover this."))
    assert result.metric_name == "refusal_correctness"
    assert result.score == 1.0
    assert result.raw == {"declines": True, "reason": "stub"}


def test_answering_an_unanswerable_question_scores_zero() -> None:
    metric = RefusalCorrectnessMetric(classifier=_classifier(False))
    assert metric.score(_unanswerable("Sure, here is how you do X...")).score == 0.0


def test_refusal_metric_does_not_apply_to_answerable_questions() -> None:
    """ValueError is the runner's signal for 'metric not applicable, skip it'."""
    metric = RefusalCorrectnessMetric(classifier=_classifier(True))
    answerable = EvalInput(question="q", answer="a", contexts=["c"], expects_refusal=False)
    with pytest.raises(ValueError, match="only applies"):
        metric.score(answerable)


def test_classifier_receives_the_question_and_answer() -> None:
    seen: list[tuple[str, str]] = []

    def spy(question: str, answer: str) -> RefusalVerdict:
        seen.append((question, answer))
        return RefusalVerdict(declines=True, reason="")

    RefusalCorrectnessMetric(classifier=spy).score(_unanswerable("nope"))
    assert seen == [("How do I do X?", "nope")]


def test_refusal_metric_needs_a_real_key_when_it_has_to_call_the_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eval_harness.metrics.refusal as refusal_module

    monkeypatch.setattr(refusal_module, "get_settings", lambda: Settings(openai_api_key="sk-..."))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        RefusalCorrectnessMetric().score(_unanswerable("The excerpts do not cover this."))


def test_answer_relevance_skips_questions_that_should_be_declined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The skip must happen BEFORE any API client is built: with an unusable key, an
    answerable question would raise RuntimeError, so a ValueError here proves the order."""
    import eval_harness.metrics.ragas_metrics as ragas_module

    monkeypatch.setattr(ragas_module, "get_settings", lambda: Settings(openai_api_key="sk-..."))
    with pytest.raises(ValueError, match="should be declined"):
        AnswerRelevanceMetric().score(_unanswerable("The excerpts do not cover this."))


def test_means_are_taken_per_category_over_the_questions_a_metric_scored() -> None:
    by_category = mean_scores_by_category(
        [
            ("direct", {"faithfulness": 1.0, "answer_relevance": 0.8}),
            ("direct", {"faithfulness": 0.5, "answer_relevance": 0.4}),
            # unanswerable questions carry no answer_relevance, so it must not be averaged in
            ("unanswerable", {"faithfulness": 1.0, "refusal_correctness": 1.0}),
        ]
    )
    assert by_category["direct"] == pytest.approx({"faithfulness": 0.75, "answer_relevance": 0.6})
    assert by_category["unanswerable"] == {"faithfulness": 1.0, "refusal_correctness": 1.0}
    assert "answer_relevance" not in by_category["unanswerable"]


def test_results_without_a_category_are_grouped_as_unknown() -> None:
    assert mean_scores_by_category([(None, {"faithfulness": 0.5})]) == {
        "unknown": {"faithfulness": 0.5}
    }
