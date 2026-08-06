from __future__ import annotations

import asyncio

from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms.base import llm_factory
from ragas.metrics.collections import AnswerRelevancy as _RagasAnswerRelevancy
from ragas.metrics.collections import ContextRecall as _RagasContextRecall
from ragas.metrics.collections import Faithfulness as _RagasFaithfulness

from eval_harness.config import get_settings
from eval_harness.metrics.base import Metric
from eval_harness.schemas import EvalInput, MetricResult

_openai_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        settings = get_settings()
        # Real OpenAI keys are far longer than this -- catches both "unset" and "still the
        # sk-... placeholder from .env.example, never replaced" with one check, rather than
        # only failing later with a confusing auth error from OpenAI's API.
        if not settings.openai_api_key or len(settings.openai_api_key) < 20:
            raise RuntimeError(
                "OPENAI_API_KEY is not set (or still the .env.example placeholder) -- "
                "required for RAGAS metrics (they use an LLM internally, separate from the "
                "GPT-4o judge in Phase 6). Add a real key to .env."
            )
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


class FaithfulnessMetric(Metric):
    """Wraps ragas' Faithfulness: are the answer's claims all supported by retrieved contexts?"""

    name = "faithfulness"

    def __init__(self, model: str | None = None):
        settings = get_settings()
        llm = llm_factory(model or settings.ragas_llm_model, client=_get_openai_client())
        self._metric = _RagasFaithfulness(llm=llm)

    def score(self, eval_input: EvalInput) -> MetricResult:
        result = asyncio.run(
            self._metric.ascore(
                user_input=eval_input.question,
                response=eval_input.answer,
                retrieved_contexts=eval_input.contexts,
            )
        )
        return MetricResult(
            metric_name=self.name, score=float(result.value), raw={"reason": result.reason}
        )


class ContextRecallMetric(Metric):
    """Wraps ragas' ContextRecall: did retrieval fetch everything needed to answer?"""

    name = "context_recall"

    def __init__(self, model: str | None = None):
        settings = get_settings()
        llm = llm_factory(model or settings.ragas_llm_model, client=_get_openai_client())
        self._metric = _RagasContextRecall(llm=llm)

    def score(self, eval_input: EvalInput) -> MetricResult:
        if not eval_input.ground_truth_answer:
            raise ValueError("context_recall requires EvalInput.ground_truth_answer")
        result = asyncio.run(
            self._metric.ascore(
                user_input=eval_input.question,
                retrieved_contexts=eval_input.contexts,
                reference=eval_input.ground_truth_answer,
            )
        )
        return MetricResult(
            metric_name=self.name, score=float(result.value), raw={"reason": result.reason}
        )


class AnswerRelevanceMetric(Metric):
    """Wraps ragas' AnswerRelevancy: does the answer actually address the question asked?"""

    name = "answer_relevance"

    def __init__(self, model: str | None = None, embedding_model: str | None = None):
        settings = get_settings()
        client = _get_openai_client()
        llm = llm_factory(model or settings.ragas_llm_model, client=client)
        embeddings = embedding_factory(
            "openai", model=embedding_model or settings.ragas_embedding_model, client=client
        )
        self._metric = _RagasAnswerRelevancy(llm=llm, embeddings=embeddings)

    def score(self, eval_input: EvalInput) -> MetricResult:
        result = asyncio.run(
            self._metric.ascore(
                user_input=eval_input.question,
                response=eval_input.answer,
            )
        )
        return MetricResult(
            metric_name=self.name, score=float(result.value), raw={"reason": result.reason}
        )
