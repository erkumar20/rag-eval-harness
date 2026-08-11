from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a meticulous RAG evaluation auditor. You are given a question, the contexts "
    "retrieved to answer it, a generated answer, and an automated metric that scored the "
    "answer poorly. Your job is to determine whether that low score is justified, and if so, "
    "explain precisely why -- name the specific failure mode (e.g. fabricated claim not "
    "supported by context, missing information the contexts didn't cover, answer doesn't "
    "address the question asked, contradicts the context) rather than giving a generic "
    "explanation. Quote the exact span of the answer or context responsible whenever "
    "possible. If, on inspection, the answer is actually fine and the low score looks wrong, "
    "say so -- your verdict does not have to agree with the metric."
)


def build_judge_prompt(
    question: str, contexts: list[str], answer: str, metric_name: str, score: float
) -> str:
    numbered_contexts = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        f"Metric: {metric_name}\n"
        f"Score: {score:.2f} (flagged as low)\n\n"
        f"Question: {question}\n\n"
        f"Retrieved contexts:\n{numbered_contexts if contexts else '(none retrieved)'}\n\n"
        f"Generated answer: {answer}\n\n"
        "Give your verdict (pass if the low score looks wrong on inspection, fail if it's "
        "justified) and a concise, specific explanation naming the failure mode."
    )
