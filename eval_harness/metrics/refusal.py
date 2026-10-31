from __future__ import annotations

import threading
from collections.abc import Callable

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from eval_harness.config import get_settings
from eval_harness.metrics.base import Metric
from eval_harness.schemas import EvalInput, MetricResult


class RefusalVerdict(BaseModel):
    declines: bool = Field(
        description=(
            "True only if the answer states that the information is not available and does "
            "NOT then supply an answer from outside knowledge."
        )
    )
    reason: str


_SYSTEM_PROMPT = (
    "You judge whether an answer DECLINES to answer a question. An answer declines when it says "
    "the provided material does not contain the information (or that it cannot answer) and stops "
    "there. If it says the information is missing but then goes on to give an answer anyway, "
    "from general knowledge, that is NOT declining. A confident, substantive answer is not "
    "declining either."
)


class RefusalCorrectnessMetric(Metric):
    """For questions the corpus cannot answer: did the pipeline decline, or did it make
    something up?

    Scored only on questions marked `expects_refusal` (the golden set's `unanswerable`
    category); on every other question it raises ValueError, which the runner treats as
    "metric does not apply" and skips. This is the check that actually targets hallucination on
    out-of-scope questions. It exists because answer_relevance cannot do the job: it scores a
    correct refusal as 0 (see devlog / open item 2), so a refusal and a failure look identical
    to it.

    Mean score is the fraction of unanswerable questions the pipeline correctly declined.
    """

    name = "refusal_correctness"

    def __init__(
        self,
        model: str | None = None,
        classifier: Callable[[str, str], RefusalVerdict] | None = None,
    ):
        self._model = model
        self._classifier = classifier
        self._lock = threading.Lock()
        self._client: instructor.Instructor | None = None

    def score(self, eval_input: EvalInput) -> MetricResult:
        if not eval_input.expects_refusal:
            raise ValueError("refusal_correctness only applies to questions the pipeline should decline")
        verdict = self._classify(eval_input.question, eval_input.answer)
        return MetricResult(
            metric_name=self.name,
            score=1.0 if verdict.declines else 0.0,
            raw={"declines": verdict.declines, "reason": verdict.reason},
        )

    def _classify(self, question: str, answer: str) -> RefusalVerdict:
        if self._classifier is not None:
            return self._classifier(question, answer)
        return self._get_client().chat.completions.create(
            model=self._model or get_settings().ragas_llm_model,
            response_model=RefusalVerdict,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
            ],
        )

    def _get_client(self) -> instructor.Instructor:
        with self._lock:
            if self._client is None:
                settings = get_settings()
                if not settings.openai_api_key or len(settings.openai_api_key) < 20:
                    raise RuntimeError(
                        "OPENAI_API_KEY is not set (or still the .env.example placeholder) -- "
                        "required for the refusal_correctness metric. Add a real key to .env."
                    )
                self._client = instructor.from_openai(OpenAI(api_key=settings.openai_api_key))
            return self._client
