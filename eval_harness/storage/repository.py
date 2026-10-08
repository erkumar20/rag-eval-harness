from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from eval_harness.schemas import EvalResult
from eval_harness.storage.models import Baseline, PipelineVersion, Result, Run


def _get_or_create_pipeline_version(session: Session, name: str, version: str) -> PipelineVersion:
    existing = session.scalar(
        select(PipelineVersion).where(
            PipelineVersion.name == name, PipelineVersion.version == version
        )
    )
    if existing is not None:
        return existing
    pipeline_version = PipelineVersion(name=name, version=version)
    session.add(pipeline_version)
    session.flush()
    return pipeline_version


def _result_row(run_id: int, eval_result: EvalResult) -> Result:
    return Result(
        run_id=run_id,
        question_id=eval_result.qa_id,
        question=eval_result.question,
        answer=eval_result.answer,
        contexts=eval_result.contexts,
        metric_scores={
            metric_result.metric_name: metric_result.score
            for metric_result in eval_result.metric_results
        },
        judge_diagnosis=(
            [diagnosis.model_dump(mode="json") for diagnosis in eval_result.judge_diagnoses]
            if eval_result.judge_diagnoses
            else None
        ),
    )


def create_run(
    session: Session,
    *,
    pipeline_name: str,
    pipeline_version: str,
    dataset_version: str | None,
) -> Run:
    """Opens a run before any question has been scored, so results can be persisted as they
    land rather than only at the end (see `save_result`)."""
    version_row = _get_or_create_pipeline_version(session, pipeline_name, pipeline_version)
    run = Run(pipeline_version=version_row, dataset_version=dataset_version)
    session.add(run)
    session.flush()
    return run


def save_result(session: Session, *, run_id: int, eval_result: EvalResult) -> None:
    """Persists one question's result into an already-open run."""
    session.add(_result_row(run_id, eval_result))
    session.flush()


def finish_run(session: Session, *, run_id: int) -> None:
    """Stamps a run as complete. A run with results but no `finished_at` is a partial run --
    one that was interrupted partway through."""
    run = session.get(Run, run_id)
    if run is not None:
        run.finished_at = datetime.now(UTC)
        session.flush()


def save_run(
    session: Session,
    *,
    pipeline_name: str,
    pipeline_version: str,
    dataset_version: str | None,
    results: list[EvalResult],
) -> Run:
    """Persists a complete run and all its results in one call.

    Convenience for callers that already hold every result (tests, backfills). The runner
    uses `create_run`/`save_result`/`finish_run` instead so an interrupted run keeps the
    questions it already paid for.
    """
    run = create_run(
        session,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        dataset_version=dataset_version,
    )
    for eval_result in results:
        session.add(_result_row(run.id, eval_result))
    run.finished_at = datetime.now(UTC)
    session.flush()
    return run


def get_latest_baseline(session: Session, *, pipeline_name: str, metric_name: str) -> float | None:
    """Returns the current baseline value for one metric, or None if no baseline has been
    set yet for this pipeline+metric -- Phase 10's gate treats that as "nothing to compare
    against, pass by default"."""
    baseline = session.scalar(
        select(Baseline).where(
            Baseline.pipeline_name == pipeline_name, Baseline.metric_name == metric_name
        )
    )
    return baseline.baseline_value if baseline is not None else None


def set_baseline(
    session: Session, *, pipeline_name: str, metric_name: str, value: float
) -> Baseline:
    """Upserts the baseline for one pipeline+metric -- relies on the unique constraint on
    (pipeline_name, metric_name) so `evalharness baseline set` always overwrites the single
    current value rather than accumulating history."""
    stmt = (
        pg_insert(Baseline)
        .values(pipeline_name=pipeline_name, metric_name=metric_name, baseline_value=value)
        .on_conflict_do_update(
            index_elements=["pipeline_name", "metric_name"],
            set_={"baseline_value": value},
        )
        .returning(Baseline)
    )
    baseline = session.scalar(stmt)
    session.flush()
    assert baseline is not None
    return baseline
