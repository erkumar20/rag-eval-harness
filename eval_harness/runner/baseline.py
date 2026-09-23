from __future__ import annotations

from dataclasses import dataclass

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


def check_baseline_gate(run_summary: RunSummary, *, threshold: float | None = None) -> GateResult:
    """Compares a run's mean scores against the stored baseline per metric. A metric with no
    baseline yet passes by default -- nothing to regress against -- so the first run for a
    new pipeline can never fail the gate on its own. `percent_drop` is positive when the
    score fell (e.g. 0.10 == a 10% drop); an improvement gives a negative value and always
    passes.
    """
    drop_threshold = threshold if threshold is not None else get_settings().baseline_drop_threshold

    metric_gates: list[MetricGateResult] = []
    with session_scope() as session:
        for metric_name, current_value in run_summary.mean_scores.items():
            baseline_value = get_latest_baseline(
                session, pipeline_name=run_summary.pipeline_name, metric_name=metric_name
            )
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

    return GateResult(passed=all(m.passed for m in metric_gates), metrics=metric_gates)


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
