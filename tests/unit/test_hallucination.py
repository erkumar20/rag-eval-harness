from __future__ import annotations

import pytest

from eval_harness.metrics.hallucination import HallucinationMetric
from eval_harness.schemas import EvalInput, StructuredAnswer, StructuredField

CONTEXTS = [
    "The invoice total is $4,500.",
    "The vendor name is Acme Corp.",
    "The delivery date is March 3rd.",
]


def make_input(fields: list[StructuredField], contexts: list[str] = CONTEXTS) -> EvalInput:
    return EvalInput(
        question="Extract the invoice details.",
        answer="(structured)",
        contexts=contexts,
        structured_answer=StructuredAnswer(fields=fields),
    )


def test_requires_structured_answer():
    metric = HallucinationMetric()
    eval_input = EvalInput(question="q", answer="a", contexts=CONTEXTS)
    with pytest.raises(ValueError, match="requires EvalInput.structured_answer"):
        metric.score(eval_input)


def test_fully_grounded_answer_scores_perfect():
    metric = HallucinationMetric()
    eval_input = make_input(
        [
            StructuredField(name="total_amount", value="$4,500", cited_chunk_ids=[0]),
            StructuredField(name="vendor_name", value="Acme Corp", cited_chunk_ids=[1]),
        ]
    )

    result = metric.score(eval_input)

    assert result.score == 1.0
    reports = result.raw["field_reports"]
    assert all(r["passed"] for r in reports)


def test_one_fabricated_field_flagged_and_nothing_else():
    metric = HallucinationMetric()
    eval_input = make_input(
        [
            StructuredField(name="total_amount", value="$4,500", cited_chunk_ids=[0]),
            StructuredField(name="vendor_name", value="Acme Corp", cited_chunk_ids=[1]),
            # Fabricated: this delivery date does not appear in the cited (or any) context.
            StructuredField(name="delivery_date", value="April 15th", cited_chunk_ids=[2]),
        ]
    )

    result = metric.score(eval_input)

    reports = {r["field_name"]: r for r in result.raw["field_reports"]}
    assert reports["total_amount"]["passed"] is True
    assert reports["vendor_name"]["passed"] is True
    assert reports["delivery_date"]["passed"] is False
    assert "not found verbatim" in reports["delivery_date"]["reason"]
    assert result.score == pytest.approx(2 / 3)


def test_one_wrong_cited_chunk_id_flagged_and_nothing_else():
    metric = HallucinationMetric()
    eval_input = make_input(
        [
            StructuredField(name="total_amount", value="$4,500", cited_chunk_ids=[0]),
            # Fabricated citation: index 99 doesn't exist (only 3 contexts, indices 0-2).
            StructuredField(name="vendor_name", value="Acme Corp", cited_chunk_ids=[99]),
        ]
    )

    result = metric.score(eval_input)

    reports = {r["field_name"]: r for r in result.raw["field_reports"]}
    assert reports["total_amount"]["passed"] is True
    assert reports["vendor_name"]["passed"] is False
    assert "fabricated citation" in reports["vendor_name"]["reason"]
    assert result.score == 0.5


def test_field_with_no_citations_fails():
    metric = HallucinationMetric()
    eval_input = make_input(
        [StructuredField(name="total_amount", value="$4,500", cited_chunk_ids=[])]
    )

    result = metric.score(eval_input)

    assert result.score == 0.0
    assert "cites no chunks" in result.raw["field_reports"][0]["reason"]


def test_match_is_case_and_whitespace_insensitive():
    metric = HallucinationMetric()
    eval_input = make_input(
        [StructuredField(name="vendor_name", value="  ACME   corp  ", cited_chunk_ids=[1])]
    )

    result = metric.score(eval_input)

    assert result.score == 1.0


def test_no_fields_is_vacuously_perfect():
    metric = HallucinationMetric()
    eval_input = make_input([])

    result = metric.score(eval_input)

    assert result.score == 1.0
    assert result.raw["field_reports"] == []


def test_negative_cited_chunk_id_is_fabricated_citation():
    metric = HallucinationMetric()
    eval_input = make_input(
        [StructuredField(name="total_amount", value="$4,500", cited_chunk_ids=[-1])]
    )

    result = metric.score(eval_input)

    assert result.score == 0.0
    assert "fabricated citation" in result.raw["field_reports"][0]["reason"]
