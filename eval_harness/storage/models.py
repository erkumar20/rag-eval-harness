from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PipelineVersion(Base):
    """One identifiable build of a RAG pipeline under test (e.g. the demo pipeline at a
    given git SHA, or the VarnueVed adapter in Phase 14) -- runs and baselines are both
    keyed off this so scores from different pipelines/versions never mix."""

    __tablename__ = "pipeline_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    version: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    runs: Mapped[list[Run]] = relationship(back_populates="pipeline_version")


class Run(Base):
    """One execution of the golden dataset against one pipeline version -- the unit that
    Phase 9's runner produces and Phase 10's gate compares against a baseline."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    pipeline_version_id: Mapped[int] = mapped_column(ForeignKey("pipeline_versions.id"))
    dataset_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    pipeline_version: Mapped[PipelineVersion] = relationship(back_populates="runs")
    results: Mapped[list[Result]] = relationship(back_populates="run")


class Result(Base):
    """One EvalResult (eval_harness/schemas.py) persisted for one question in one run --
    mirrors that Pydantic model's shape so save/load is a straight field mapping."""

    __tablename__ = "results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    question_id: Mapped[str] = mapped_column(String(255))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    contexts: Mapped[list] = mapped_column(JSONB)
    metric_scores: Mapped[dict] = mapped_column(JSONB)
    # none_as_null=True, otherwise SQLAlchemy stores an unjudged question's Python None as
    # JSON null rather than SQL NULL, and `WHERE judge_diagnosis IS NOT NULL` then silently
    # matches every row -- which made a first real run look like all 35 questions had been
    # judged when only 25 actually were.
    judge_diagnosis: Mapped[list | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped[Run] = relationship(back_populates="results")


class Baseline(Base):
    """The stored reference value each metric is compared against for regression gating
    (Phase 10) -- one row per (pipeline_name, metric_name), overwritten by
    `evalharness baseline set`."""

    __tablename__ = "baselines"
    __table_args__ = (
        # One current baseline per pipeline+metric -- setting a new one replaces the row
        # rather than accumulating history, since only the latest baseline is ever compared
        # against.
        UniqueConstraint("pipeline_name", "metric_name", name="uq_baseline_pipeline_metric"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    pipeline_name: Mapped[str] = mapped_column(String(255), index=True)
    metric_name: Mapped[str] = mapped_column(String(255))
    baseline_value: Mapped[float] = mapped_column(Float)
    set_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
