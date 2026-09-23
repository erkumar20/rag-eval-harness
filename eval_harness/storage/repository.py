from __future__ import annotations

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


def save_run(
    session: Session,
    *,
    pipeline_name: str,
    pipeline_version: str,
    dataset_version: str | None,
    results: list[EvalResult],
) -> Run:
    """Persists one full evaluation run and all its per-question results. Called by
    Phase 9's runner after every question in the golden set has been scored."""
    version_row = _get_or_create_pipeline_version(session, pipeline_name, pipeline_version)
    run = Run(pipeline_version=version_row, dataset_version=dataset_version)
    session.add(run)
    session.flush()

    for eval_result in results:
        session.add(
            Result(
                run_id=run.id,
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
        )

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
