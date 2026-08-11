"""Phase 5 sanity check: score manufactured examples with known expected directions before
trusting the RAGAS metric wrappers on the real golden dataset.

Not a pytest suite -- these are live LLM calls (real cost, real latency), meant to be run
manually/occasionally, not on every test run. Re-run any time the underlying ragas version,
model, or metric wrapper implementation changes.
"""

from __future__ import annotations

from eval_harness.metrics.ragas_metrics import (
    AnswerRelevanceMetric,
    ContextRecallMetric,
    FaithfulnessMetric,
)
from eval_harness.schemas import EvalInput

BG_TASK_CONTEXTS = [
    "First, import BackgroundTasks and define a parameter in your path operation function "
    "with a type declaration of BackgroundTasks.",
    "FastAPI will create the object of type BackgroundTasks for you and pass it as that "
    "parameter.",
    'background_tasks.add_task(write_notification, email, message="some notification")',
]
BG_TASK_GROUND_TRUTH = (
    "Declare a BackgroundTasks parameter in your path operation function, then call "
    "background_tasks.add_task(func, *args) to run it after returning a response."
)

CASES = [
    {
        "label": "A: grounded + relevant + good retrieval (expect HIGH on all three)",
        "input": EvalInput(
            question="How do I add a background task in FastAPI?",
            answer=(
                "Declare a parameter of type BackgroundTasks in your path operation function. "
                "FastAPI creates that object for you and passes it as the parameter. You can "
                "then call background_tasks.add_task() to add a background task."
            ),
            contexts=BG_TASK_CONTEXTS,
            ground_truth_answer=BG_TASK_GROUND_TRUTH,
        ),
        "expect": "faithfulness HIGH, answer_relevance HIGH, context_recall HIGH",
    },
    {
        "label": "B: hallucinated claims added on top of real grounding (expect LOW faithfulness)",
        "input": EvalInput(
            question="How do I add a background task in FastAPI?",
            answer=(
                "Declare a parameter of type BackgroundTasks, then call "
                "background_tasks.add_task(...). Background tasks in FastAPI automatically "
                "retry up to 3 times on failure and are persisted to a Redis queue for "
                "durability."
            ),
            contexts=BG_TASK_CONTEXTS,
            ground_truth_answer=BG_TASK_GROUND_TRUTH,
        ),
        "expect": "faithfulness LOW (retry/Redis claims are fabricated, not in context)",
    },
    {
        "label": "C: faithful but tangential -- doesn't actually answer the question (expect LOWER answer_relevance)",
        "input": EvalInput(
            question="How do I add a background task in FastAPI?",
            answer=(
                "FastAPI automatically creates the BackgroundTasks object for you and passes "
                "it as a parameter."
            ),
            contexts=BG_TASK_CONTEXTS,
            ground_truth_answer=BG_TASK_GROUND_TRUTH,
        ),
        "expect": "answer_relevance LOWER than case A (true statement, doesn't explain how-to)",
    },
    {
        "label": "D: bad retrieval -- contexts don't contain what's needed (expect LOW context_recall)",
        "input": EvalInput(
            question="How do I add a background task in FastAPI?",
            answer="Not enough information retrieved to answer confidently.",
            contexts=[
                "FastAPI is a modern, fast web framework for building APIs with Python.",
                "You can install FastAPI using pip install fastapi.",
            ],
            ground_truth_answer=BG_TASK_GROUND_TRUTH,
        ),
        "expect": "context_recall LOW (retrieved contexts are real but off-topic)",
    },
    {
        "label": "E: second clean positive control, different topic (expect HIGH on all three)",
        "input": EvalInput(
            question="How do I make a FastAPI query parameter required?",
            answer=(
                "Just don't declare a default value for the parameter -- without a default, "
                "FastAPI treats it as required."
            ),
            contexts=[
                "when you want to make a query parameter required, you can just not declare "
                "any default value",
                "Here the query parameter `needy` is a required query parameter of type `str`.",
            ],
            ground_truth_answer=(
                "Don't give the parameter a default value; without one, FastAPI treats it as "
                "required."
            ),
        ),
        "expect": "faithfulness HIGH, answer_relevance HIGH, context_recall HIGH",
    },
]


def main() -> None:
    faithfulness = FaithfulnessMetric()
    context_recall = ContextRecallMetric()
    answer_relevance = AnswerRelevanceMetric()

    for case in CASES:
        eval_input = case["input"]
        f_score = faithfulness.score(eval_input)
        cr_score = context_recall.score(eval_input)
        ar_score = answer_relevance.score(eval_input)

        print(f"\n{case['label']}")
        print(f"  expect: {case['expect']}")
        print(f"  faithfulness   = {f_score.score:.3f}")
        print(f"  context_recall = {cr_score.score:.3f}")
        print(f"  answer_relevance = {ar_score.score:.3f}")


if __name__ == "__main__":
    main()
