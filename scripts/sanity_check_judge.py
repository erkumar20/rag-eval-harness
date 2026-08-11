"""Phase 6 sanity check: manufactured failure cases, run through the real Phase 5 metrics
first, then the judge only if the score falls below the configured threshold -- demonstrating
the actual intended cost-control workflow (Phase 6 plan step 4), not just calling the judge
unconditionally.

Live LLM calls, real cost -- meant to be run manually/occasionally, not in the test suite.
"""

from __future__ import annotations

from eval_harness.config import get_settings
from eval_harness.judge.gpt4o_judge import GPT4oJudge
from eval_harness.metrics.base import Metric
from eval_harness.metrics.ragas_metrics import (
    AnswerRelevanceMetric,
    ContextRecallMetric,
    FaithfulnessMetric,
)
from eval_harness.schemas import EvalInput

BG_TASK_CONTEXTS = [
    (
        "First, import BackgroundTasks and define a parameter in your path operation function "
        "with a type declaration of BackgroundTasks."
    ),
    (
        "FastAPI will create the object of type BackgroundTasks for you and pass it as that "
        "parameter."
    ),
    'background_tasks.add_task(write_notification, email, message="some notification")',
]

CASES: list[dict] = [
    {
        "label": "1: fabricated retry/Redis claims on top of real grounding (from Phase 5)",
        "metric_cls": FaithfulnessMetric,
        "input": EvalInput(
            question="How do I add a background task in FastAPI?",
            answer=(
                "Declare a parameter of type BackgroundTasks, then call "
                "background_tasks.add_task(...). Background tasks in FastAPI automatically "
                "retry up to 3 times on failure and are persisted to a Redis queue for "
                "durability."
            ),
            contexts=BG_TASK_CONTEXTS,
        ),
    },
    {
        "label": "2: fabricated pricing (the plan's own illustrative example)",
        "metric_cls": FaithfulnessMetric,
        "input": EvalInput(
            question="What is the price of the Pro plan?",
            answer="The Pro plan costs $49 per month.",
            contexts=[
                (
                    "The Pro plan includes unlimited projects, priority support, and advanced "
                    "analytics."
                ),
                "Pricing details are available upon request from our sales team.",
            ],
        ),
    },
    {
        "label": "3: off-topic answer (relevance failure)",
        "metric_cls": AnswerRelevanceMetric,
        "input": EvalInput(
            question="How do I add a background task in FastAPI?",
            answer=(
                "FastAPI is a modern, high-performance web framework for building APIs with "
                "Python, based on standard Python type hints."
            ),
            contexts=BG_TASK_CONTEXTS,
        ),
    },
    {
        "label": "4: real retrieval gap found in Phase 5 (q005, ge=1 numeric validation)",
        "metric_cls": ContextRecallMetric,
        "input": EvalInput(
            question="What does setting ge=1 on a path parameter's Path() validation mean?",
            answer=(
                "Unfortunately, the provided excerpts do not contain information about "
                "setting ge=1 on a path parameter's Path() validation."
            ),
            contexts=[
                (
                    'Path Parameters -- You can declare path "parameters" or "variables" with '
                    "the same syntax used by Python format strings."
                ),
                (
                    "Query Parameters -- When you declare other function parameters that are "
                    "not part of the path parameters, they are automatically interpreted as "
                    '"query" parameters.'
                ),
            ],
            ground_truth_answer=(
                "It means the value must be an integer greater than or equal to 1 -- 'ge' "
                "stands for 'greater than or equal'."
            ),
        ),
    },
]


def main() -> None:
    threshold = get_settings().judge_score_threshold
    judge = GPT4oJudge()

    for case in CASES:
        metric: Metric = case["metric_cls"]()
        eval_input: EvalInput = case["input"]
        result = metric.score(eval_input)

        print(f"\n{case['label']}")
        print(f"  {metric.name} = {result.score:.3f} (threshold = {threshold})")

        if result.score >= threshold:
            print("  -> above threshold, judge NOT invoked (this is the cost control working)")
            continue

        diagnosis = judge.judge(
            question=eval_input.question,
            contexts=eval_input.contexts,
            answer=eval_input.answer,
            metric_name=metric.name,
            score=result.score,
        )
        print(f"  judge verdict: {diagnosis.verdict}")
        print(f"  judge explanation: {diagnosis.explanation}")
        if diagnosis.evidence_span:
            print(f"  evidence span: {diagnosis.evidence_span!r}")


if __name__ == "__main__":
    main()
