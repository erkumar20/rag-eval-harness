"""Integration tests against the real local Postgres instance (Phase 8 step 6). Not mocked
-- these exercise actual DDL/DML against `rag_eval`, unlike the mocked unit tests elsewhere."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from eval_harness.judge.schema import Diagnosis
from eval_harness.schemas import EvalResult, MetricResult
from eval_harness.storage.db import session_scope
from eval_harness.storage.models import PipelineVersion, Result, Run
from eval_harness.storage.repository import get_latest_baseline, save_run, set_baseline


@pytest.fixture
def sample_results() -> list[EvalResult]:
    return [
        EvalResult(
            qa_id="q001",
            question="What decorator marks a path operation as GET?",
            answer="@app.get(...)",
            contexts=["FastAPI uses @app.get() to declare a GET path operation."],
            pipeline_version="test-v1",
            dataset_version="v1",
            metric_results=[
                MetricResult(metric_name="faithfulness", score=1.0),
                MetricResult(metric_name="context_recall", score=0.9),
            ],
            judge_diagnoses=[],
        ),
        EvalResult(
            qa_id="q002",
            question="Does FastAPI support automatic retries on Redis?",
            answer="Yes, FastAPI automatically retries failed Redis calls.",
            contexts=["FastAPI has no built-in Redis integration."],
            pipeline_version="test-v1",
            dataset_version="v1",
            metric_results=[MetricResult(metric_name="faithfulness", score=0.0)],
            judge_diagnoses=[
                Diagnosis(
                    metric="faithfulness",
                    verdict="fail",
                    explanation="Fabricated a Redis retry claim not present in context.",
                    evidence_span="automatically retries failed Redis calls",
                )
            ],
        ),
    ]


def test_save_run_persists_results_and_diagnoses(sample_results: list[EvalResult]) -> None:
    with session_scope() as session:
        run = save_run(
            session,
            pipeline_name="test-pipeline",
            pipeline_version="test-v1",
            dataset_version="v1",
            results=sample_results,
        )
        run_id = run.id

    with session_scope() as session:
        persisted_run = session.get(Run, run_id)
        assert persisted_run is not None
        assert persisted_run.dataset_version == "v1"

        pipeline_version = session.get(PipelineVersion, persisted_run.pipeline_version_id)
        assert pipeline_version is not None
        assert pipeline_version.name == "test-pipeline"

        results = session.scalars(select(Result).where(Result.run_id == run_id)).all()
        assert len(results) == 2

        by_question_id = {r.question_id: r for r in results}
        assert by_question_id["q001"].metric_scores == {
            "faithfulness": 1.0,
            "context_recall": 0.9,
        }
        assert by_question_id["q001"].judge_diagnosis is None

        q002 = by_question_id["q002"]
        assert q002.metric_scores == {"faithfulness": 0.0}
        assert q002.judge_diagnosis is not None
        assert q002.judge_diagnosis[0]["verdict"] == "fail"
        assert "Redis" in q002.judge_diagnosis[0]["explanation"]


def test_save_run_reuses_existing_pipeline_version(sample_results: list[EvalResult]) -> None:
    with session_scope() as session:
        run_a = save_run(
            session,
            pipeline_name="test-pipeline-reuse",
            pipeline_version="v1",
            dataset_version="v1",
            results=sample_results[:1],
        )
        run_b = save_run(
            session,
            pipeline_name="test-pipeline-reuse",
            pipeline_version="v1",
            dataset_version="v1",
            results=sample_results[:1],
        )
        assert run_a.pipeline_version_id == run_b.pipeline_version_id


def test_set_baseline_then_get_latest_baseline_roundtrips() -> None:
    with session_scope() as session:
        set_baseline(
            session, pipeline_name="test-baseline-pipeline", metric_name="faithfulness", value=0.9
        )
        value = get_latest_baseline(
            session, pipeline_name="test-baseline-pipeline", metric_name="faithfulness"
        )
        assert value == 0.9


def test_set_baseline_overwrites_rather_than_duplicates() -> None:
    with session_scope() as session:
        set_baseline(
            session, pipeline_name="test-baseline-overwrite", metric_name="faithfulness", value=0.8
        )
        set_baseline(
            session, pipeline_name="test-baseline-overwrite", metric_name="faithfulness", value=0.95
        )
        value = get_latest_baseline(
            session, pipeline_name="test-baseline-overwrite", metric_name="faithfulness"
        )
        assert value == 0.95


def test_get_latest_baseline_returns_none_when_unset() -> None:
    with session_scope() as session:
        value = get_latest_baseline(
            session, pipeline_name="never-baselined-pipeline", metric_name="faithfulness"
        )
        assert value is None
