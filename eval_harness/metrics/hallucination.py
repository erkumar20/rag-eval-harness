from __future__ import annotations

import re

from pydantic import BaseModel

from eval_harness.metrics.base import Metric
from eval_harness.schemas import EvalInput, MetricResult, StructuredField


class FieldCheckResult(BaseModel):
    field_name: str
    passed: bool
    reason: str


class HallucinationMetric(Metric):
    """Field-level grounding check for structured extraction answers.

    Unlike RAGAS faithfulness (a single aggregate ratio over whole-sentence claims), this
    checks each field of a structured answer independently against the specific contexts it
    cites, and reports a per-field pass/fail -- the granularity RAGAS's sentence-level NLI
    approach doesn't give you (see devlog, Phase 5: the aggregate ratio is sensitive to
    statement decomposition/phrasing in ways that can obscure exactly which claim is wrong).

    Two failure modes are checked, independently, per field:
    1. Fabricated citation: a cited_chunk_ids entry doesn't correspond to any retrieved
       context at all (out of range).
    2. Ungrounded value: the field's value doesn't appear (case/whitespace-insensitive) in
       any of the contexts it specifically cites -- citing a real chunk whose content doesn't
       actually support the claimed value.

    Deliberately strict, verbatim-only matching -- no semantic-similarity fallback. An earlier
    version tried one (see devlog, Phase 7): empirically, embedding similarity could not
    reliably separate legitimate paraphrases from fabricated same-topic claims with a swapped
    number or name -- the fabrications sometimes scored *higher* than real paraphrases, since
    embeddings are far more sensitive to topic than to a specific factual detail. That's the
    exact failure mode a hallucination detector most needs to catch, so a lenient match would
    have defeated the metric's purpose. Verbatim matching is also the *correct* expectation
    for this metric's target use case: structured field extraction (e.g. an invoice's
    total_amount, vendor_name), where an atomic extracted value should appear near-verbatim in
    the source, not be a paraphrase. Free-text answer faithfulness is RAGAS's job (Phase 5),
    not this metric's.

    No live LLM calls -- purely string-based, which also makes it far cheaper to run than the
    RAGAS metrics or the judge.
    """

    name = "hallucination"

    def score(self, eval_input: EvalInput) -> MetricResult:
        if eval_input.structured_answer is None:
            raise ValueError("hallucination metric requires EvalInput.structured_answer")

        contexts = eval_input.contexts
        reports = [
            self._check_field(field, contexts) for field in eval_input.structured_answer.fields
        ]

        total = len(reports)
        score = sum(1 for r in reports if r.passed) / total if total else 1.0

        return MetricResult(
            metric_name=self.name,
            score=score,
            raw={"field_reports": [r.model_dump() for r in reports]},
        )

    def _check_field(self, field: StructuredField, contexts: list[str]) -> FieldCheckResult:
        invalid_ids = [cid for cid in field.cited_chunk_ids if cid < 0 or cid >= len(contexts)]
        if invalid_ids:
            return FieldCheckResult(
                field_name=field.name,
                passed=False,
                reason=(
                    f"cited_chunk_ids {invalid_ids} do not correspond to any retrieved "
                    "context (fabricated citation)"
                ),
            )

        if not field.cited_chunk_ids:
            return FieldCheckResult(
                field_name=field.name,
                passed=False,
                reason="field cites no chunks -- cannot verify grounding",
            )

        cited_texts = [contexts[cid] for cid in field.cited_chunk_ids]
        if self._verbatim_match(field.value, cited_texts):
            return FieldCheckResult(
                field_name=field.name, passed=True, reason="value found verbatim in cited context"
            )

        return FieldCheckResult(
            field_name=field.name,
            passed=False,
            reason=f"value {field.value!r} not found verbatim in cited contexts",
        )

    @staticmethod
    def _verbatim_match(value: str, texts: list[str]) -> bool:
        normalized_value = _normalize_whitespace(value)
        if not normalized_value:
            return False
        return any(normalized_value in _normalize_whitespace(text) for text in texts)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
