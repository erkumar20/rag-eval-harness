from __future__ import annotations

import pytest

from eval_harness.schemas import EvalInput

# These tests deliberately avoid any live LLM call -- they only exercise the wrapper's own
# input validation and error paths. Live scoring is exercised separately (see devlog: Phase 5
# sanity test against manufactured examples, run manually once OPENAI_API_KEY is available).


def test_context_recall_requires_ground_truth_answer(monkeypatch: pytest.MonkeyPatch):
    import eval_harness.metrics.ragas_metrics as ragas_metrics_module
    from eval_harness.config import Settings

    monkeypatch.setattr(
        ragas_metrics_module,
        "get_settings",
        lambda: Settings(openai_api_key="sk-test-not-a-real-key-but-long-enough-1234"),
    )

    metric = ragas_metrics_module.ContextRecallMetric()
    eval_input = EvalInput(question="q", answer="a", contexts=["c"], ground_truth_answer=None)

    with pytest.raises(ValueError, match="requires EvalInput.ground_truth_answer"):
        metric.score(eval_input)


@pytest.mark.parametrize("bad_key", [None, "", "sk-...", "short"])
def test_metrics_raise_clear_error_without_a_real_api_key(
    monkeypatch: pytest.MonkeyPatch, bad_key: str | None
):
    import eval_harness.metrics.ragas_metrics as ragas_metrics_module
    from eval_harness.config import Settings

    monkeypatch.setattr(
        ragas_metrics_module, "get_settings", lambda: Settings(openai_api_key=bad_key)
    )
    ragas_metrics_module._openai_client = None  # reset the lazily-initialized singleton

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        ragas_metrics_module.FaithfulnessMetric()


def test_metric_names():
    from eval_harness.metrics.ragas_metrics import (
        AnswerRelevanceMetric,
        ContextRecallMetric,
        FaithfulnessMetric,
    )

    assert FaithfulnessMetric.name == "faithfulness"
    assert ContextRecallMetric.name == "context_recall"
    assert AnswerRelevanceMetric.name == "answer_relevance"
