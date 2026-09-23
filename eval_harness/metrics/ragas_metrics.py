from __future__ import annotations

import asyncio
import threading

from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms.base import llm_factory
from ragas.metrics.collections import AnswerRelevancy as _RagasAnswerRelevancy
from ragas.metrics.collections import ContextRecall as _RagasContextRecall
from ragas.metrics.collections import Faithfulness as _RagasFaithfulness

from eval_harness.config import get_settings
from eval_harness.metrics.base import Metric
from eval_harness.schemas import EvalInput, MetricResult

# ragas' llm_factory defaults to max_tokens=1024, which truncates structured verdict output
# (one entry per atomic statement, each with a reason string) on longer answers/contexts --
# hit a real IncompleteOutputException at q011 of the golden set with the default. 4096 gives
# enough headroom for the longest realistic answer in this corpus.
_RAGAS_LLM_MAX_TOKENS = 4096

_thread_local = threading.local()


def _get_openai_client() -> AsyncOpenAI:
    """One AsyncOpenAI client per thread, not a process-wide singleton.

    The Phase 9 runner scores questions concurrently across a thread pool, and each call
    into a metric's `score()` does `asyncio.run(...)`, which spins up its own event loop.
    AsyncOpenAI's underlying httpx client binds internal locks/connection-pool state to
    whichever event loop it's first used on -- sharing one client instance across threads
    (each running a *different* event loop) doesn't raise a clean error, it just silently
    thrashes: a real live run of the golden set was still running after 24+ minutes under
    5-way concurrency, versus ~13s/question (well under the sequential total) with a
    thread-local client. See devlog/challenges_and_fixes.md.
    """
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
    if not hasattr(_thread_local, "openai_client"):
        _thread_local.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _thread_local.openai_client


class FaithfulnessMetric(Metric):
    """Wraps ragas' Faithfulness: are the answer's claims all supported by retrieved contexts?

    The underlying ragas metric object (and the OpenAI client it wraps) is built lazily, once
    per thread, on first use -- not eagerly in __init__ -- for the same cross-event-loop
    reason as `_get_openai_client()` above. Building it in __init__ would bind it to whatever
    thread constructs this object (typically the CLI's main thread), then every worker thread
    would reuse that same instance regardless.
    """

    name = "faithfulness"

    def __init__(self, model: str | None = None):
        self._model = model
        self._thread_local = threading.local()

    def _get_metric(self) -> _RagasFaithfulness:
        if not hasattr(self._thread_local, "metric"):
            settings = get_settings()
            llm = llm_factory(
                self._model or settings.ragas_llm_model,
                client=_get_openai_client(),
                max_tokens=_RAGAS_LLM_MAX_TOKENS,
            )
            self._thread_local.metric = _RagasFaithfulness(llm=llm)
        return self._thread_local.metric

    def score(self, eval_input: EvalInput) -> MetricResult:
        result = asyncio.run(
            self._get_metric().ascore(
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
        self._model = model
        self._thread_local = threading.local()

    def _get_metric(self) -> _RagasContextRecall:
        if not hasattr(self._thread_local, "metric"):
            settings = get_settings()
            llm = llm_factory(
                self._model or settings.ragas_llm_model,
                client=_get_openai_client(),
                max_tokens=_RAGAS_LLM_MAX_TOKENS,
            )
            self._thread_local.metric = _RagasContextRecall(llm=llm)
        return self._thread_local.metric

    def score(self, eval_input: EvalInput) -> MetricResult:
        if not eval_input.ground_truth_answer:
            raise ValueError("context_recall requires EvalInput.ground_truth_answer")
        result = asyncio.run(
            self._get_metric().ascore(
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
        self._model = model
        self._embedding_model = embedding_model
        self._thread_local = threading.local()

    def _get_metric(self) -> _RagasAnswerRelevancy:
        if not hasattr(self._thread_local, "metric"):
            settings = get_settings()
            client = _get_openai_client()
            llm = llm_factory(
                self._model or settings.ragas_llm_model,
                client=client,
                max_tokens=_RAGAS_LLM_MAX_TOKENS,
            )
            embeddings = embedding_factory(
                "openai", model=self._embedding_model or settings.ragas_embedding_model, client=client
            )
            self._thread_local.metric = _RagasAnswerRelevancy(llm=llm, embeddings=embeddings)
        return self._thread_local.metric

    def score(self, eval_input: EvalInput) -> MetricResult:
        result = asyncio.run(
            self._get_metric().ascore(
                user_input=eval_input.question,
                response=eval_input.answer,
            )
        )
        return MetricResult(
            metric_name=self.name, score=float(result.value), raw={"reason": result.reason}
        )
