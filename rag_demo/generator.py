from __future__ import annotations

from groq import Groq

from eval_harness.config import get_settings

DEFAULT_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about FastAPI using only the provided "
    "documentation excerpts. Answer using only information present in the excerpts below. "
    "If the excerpts don't contain enough information to answer, say so explicitly rather "
    "than guessing or using outside knowledge."
)


def build_prompt(question: str, contexts: list[str]) -> str:
    numbered_contexts = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        f"Documentation excerpts:\n{numbered_contexts}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely, using only the excerpts above."
    )


_client: Groq | None = None


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set -- add it to .env before running the demo pipeline."
            )
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def generate_answer(question: str, contexts: list[str], model: str = DEFAULT_MODEL) -> str:
    client = get_groq_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, contexts)},
        ],
        # Deterministic output so repeated eval runs are comparable (same question+contexts
        # in -> same answer out), which matters for regression gating in Phase 10.
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()
