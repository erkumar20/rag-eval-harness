from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eval_harness.config import get_settings
from eval_harness.runner.evaluate import RunSummary
from eval_harness.storage.db import session_scope
from eval_harness.storage.repository import get_latest_baseline, set_baseline


@dataclass
class MetricGateResult:
    metric_name: str
    current_value: float
    baseline_value: float | None
    percent_drop: float | None
    passed: bool


@dataclass
class GateResult:
    passed: bool
    metrics: list[MetricGateResult]
    baseline_dataset_version: str | None = None
    run_dataset_version: str | None = None

    @property
    def dataset_changed(self) -> bool:
        """True when the baseline was recorded against a different version of the golden set,
        which makes the comparison apples-to-oranges: a score move may be the dataset
        changing rather than the pipeline regressing. This is exactly what Phase 4's content
        hash of the dataset file exists to detect."""
        return (
            self.baseline_dataset_version is not None
            and self.run_dataset_version is not None
            and self.baseline_dataset_version != self.run_dataset_version
        )


def _baselines_from_db(pipeline_name: str, metric_names: list[str]) -> dict[str, float]:
    baselines: dict[str, float] = {}
    with session_scope() as session:
        for metric_name in metric_names:
            value = get_latest_baseline(
                session, pipeline_name=pipeline_name, metric_name=metric_name
            )
            if value is not None:
                baselines[metric_name] = value
    return baselines


def load_baseline_file(path: Path) -> tuple[dict[str, float], str | None]:
    """Reads a committed baseline snapshot. Returns (metric -> value, dataset_version)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["metrics"], data.get("dataset_version")


def write_baseline_file(run_summary: RunSummary, path: Path) -> None:
    """Writes the run's mean scores as a committed baseline snapshot.

    CI gets a fresh, empty Postgres on every job, so a database-only baseline would always
    read back as "no baseline yet" and the gate would silently pass every time. Committing
    the baseline instead keeps CI self-contained *and* makes any change to it show up in the
    PR diff -- so a PR that lowers the bar to make itself pass is visible to a reviewer.
    """
    payload = {
        "pipeline_name": run_summary.pipeline_name,
        "pipeline_version": run_summary.pipeline_version,
        "dataset_version": run_summary.dataset_version,
        "recorded_at": datetime.now(UTC).isoformat(),
        "metrics": run_summary.mean_scores,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def check_baseline_gate(
    run_summary: RunSummary,
    *,
    threshold: float | None = None,
    baseline_file: Path | None = None,
) -> GateResult:
    """Compares a run's mean scores against the baseline per metric.

    Reads the baseline from `baseline_file` when given (how CI runs, since its Postgres is
    ephemeral), otherwise from the database. A metric with no baseline yet passes by default
    -- nothing to regress against -- so a pipeline's first run can never fail its own gate.
    `percent_drop` is positive when the score fell (0.10 == a 10% drop); an improvement gives
    a negative value and always passes.
    """
    drop_threshold = threshold if threshold is not None else get_settings().baseline_drop_threshold
    metric_names = list(run_summary.mean_scores)

    baseline_dataset_version: str | None = None
    if baseline_file is not None:
        baselines, baseline_dataset_version = load_baseline_file(baseline_file)
    else:
        baselines = _baselines_from_db(run_summary.pipeline_name, metric_names)

    metric_gates: list[MetricGateResult] = []
    for metric_name, current_value in run_summary.mean_scores.items():
        baseline_value = baselines.get(metric_name)
        if baseline_value is None:
            metric_gates.append(
                MetricGateResult(metric_name, current_value, None, None, passed=True)
            )
            continue

        percent_drop = (
            (baseline_value - current_value) / baseline_value if baseline_value != 0 else 0.0
        )
        metric_gates.append(
            MetricGateResult(
                metric_name,
                current_value,
                baseline_value,
                percent_drop,
                passed=percent_drop <= drop_threshold,
            )
        )

    return GateResult(
        passed=all(m.passed for m in metric_gates),
        metrics=metric_gates,
        baseline_dataset_version=baseline_dataset_version,
        run_dataset_version=run_summary.dataset_version,
    )


def set_baseline_from_run(run_summary: RunSummary) -> None:
    """Establishes the current run's mean scores as the new baseline for its pipeline --
    what `evalharness baseline set` calls."""
    with session_scope() as session:
        for metric_name, value in run_summary.mean_scores.items():
            set_baseline(
                session,
                pipeline_name=run_summary.pipeline_name,
                metric_name=metric_name,
                value=value,
            )
