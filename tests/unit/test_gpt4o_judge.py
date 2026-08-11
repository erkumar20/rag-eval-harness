from __future__ import annotations

import pytest

from eval_harness.judge.prompts import build_judge_prompt

# Live judge calls are exercised separately (devlog: Phase 6 sanity test against manufactured
# failures, run manually). These tests cover the parts that don't need a live LLM call.


@pytest.mark.parametrize("bad_key", [None, "", "sk-...", "short"])
def test_judge_raises_clear_error_without_a_real_api_key(
    monkeypatch: pytest.MonkeyPatch, bad_key: str | None
):
    import eval_harness.judge.gpt4o_judge as judge_module
    from eval_harness.config import Settings

    monkeypatch.setattr(judge_module, "get_settings", lambda: Settings(openai_api_key=bad_key))

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        judge_module.GPT4oJudge()


def test_judge_defaults_to_configured_judge_model(monkeypatch: pytest.MonkeyPatch):
    import eval_harness.judge.gpt4o_judge as judge_module
    from eval_harness.config import Settings

    monkeypatch.setattr(
        judge_module,
        "get_settings",
        lambda: Settings(openai_api_key="sk-test-not-real-but-long-enough-123", judge_model="gpt-4o"),
    )

    judge = judge_module.GPT4oJudge()
    assert judge._model == "gpt-4o"


def test_judge_constructor_model_overrides_settings(monkeypatch: pytest.MonkeyPatch):
    import eval_harness.judge.gpt4o_judge as judge_module
    from eval_harness.config import Settings

    monkeypatch.setattr(
        judge_module,
        "get_settings",
        lambda: Settings(openai_api_key="sk-test-not-real-but-long-enough-123", judge_model="gpt-4o"),
    )

    judge = judge_module.GPT4oJudge(model="gpt-4o-mini")
    assert judge._model == "gpt-4o-mini"


def test_build_judge_prompt_includes_all_inputs():
    prompt = build_judge_prompt(
        question="How do I add a background task?",
        contexts=["ctx one", "ctx two"],
        answer="Some answer with a fabricated claim.",
        metric_name="faithfulness",
        score=0.25,
    )
    assert "How do I add a background task?" in prompt
    assert "ctx one" in prompt
    assert "ctx two" in prompt
    assert "Some answer with a fabricated claim." in prompt
    assert "faithfulness" in prompt
    assert "0.25" in prompt


def test_build_judge_prompt_handles_empty_contexts():
    prompt = build_judge_prompt(
        question="q", contexts=[], answer="a", metric_name="context_recall", score=0.0
    )
    assert "(none retrieved)" in prompt
