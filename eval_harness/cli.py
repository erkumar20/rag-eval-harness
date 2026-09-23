from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from eval_harness.config import get_settings
from eval_harness.dataset.loader import DEFAULT_DATASET_PATH, dataset_version, load_golden_dataset
from eval_harness.judge.gpt4o_judge import GPT4oJudge
from eval_harness.metrics.base import Metric
from eval_harness.metrics.ragas_metrics import (
    AnswerRelevanceMetric,
    ContextRecallMetric,
    FaithfulnessMetric,
)
from eval_harness.runner.baseline import (
    GateResult,
    check_baseline_gate,
    set_baseline_from_run,
    write_baseline_file,
)
from eval_harness.runner.evaluate import RunSummary, run_evaluation
from eval_harness.storage.db import session_scope
from eval_harness.storage.models import PipelineVersion, Result, Run

app = typer.Typer(help="RAG pipeline evaluation harness: run, gate, and report.")
console = Console()


def _default_metrics() -> list[Metric]:
    # The three RAGAS metrics (Phase 5), which apply to any pipeline producing plain
    # answer/contexts. HallucinationMetric (Phase 7) is opt-in per pipeline -- it needs
    # EvalInput.structured_answer, which the demo pipeline never populates -- so it isn't
    # included here; a structured-output pipeline's own run script adds it explicitly.
    return [FaithfulnessMetric(), ContextRecallMetric(), AnswerRelevanceMetric()]


def _load_pipeline(name: str):
    if name == "demo":
        from rag_demo.pipeline import DemoRAGPipeline

        return DemoRAGPipeline()
    raise typer.BadParameter(f"Unknown pipeline {name!r}. Registered pipelines: demo")


def _summary_from_run(run_id: int) -> RunSummary:
    """Rebuilds a RunSummary from a persisted run, so the baseline commands can work off a
    run id rather than only off a run that just happened in-process."""
    with session_scope() as session:
        db_run = session.get(Run, run_id)
        if db_run is None:
            console.print(f"[red]No run with id {run_id}.[/red]")
            raise typer.Exit(code=1)
        pipeline_version = session.get(PipelineVersion, db_run.pipeline_version_id)
        results = session.scalars(select(Result).where(Result.run_id == run_id)).all()

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for result in results:
            for metric_name, score in result.metric_scores.items():
                sums[metric_name] = sums.get(metric_name, 0.0) + score
                counts[metric_name] = counts.get(metric_name, 0) + 1

        return RunSummary(
            run_id=run_id,
            pipeline_name=pipeline_version.name,
            pipeline_version=pipeline_version.version,
            dataset_version=db_run.dataset_version,
            mean_scores={name: sums[name] / counts[name] for name in sums},
        )


def _gate_markdown(summary: RunSummary, gate_result: GateResult) -> str:
    """Markdown table of metric deltas, posted by CI as a PR comment."""
    verdict = "✅ **Gate passed**" if gate_result.passed else "❌ **Gate FAILED**"
    threshold = get_settings().baseline_drop_threshold

    lines = [
        f"## RAG eval gate — `{summary.pipeline_name}`",
        "",
        verdict,
        "",
        "| Metric | Baseline | Current | Change | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for gate in gate_result.metrics:
        if gate.baseline_value is None:
            lines.append(f"| `{gate.metric_name}` | — | {gate.current_value:.3f} | — | no baseline |")
            continue
        # percent_drop is positive when the score fell, so negate it to read as a delta.
        status = "pass" if gate.passed else "**FAIL**"
        lines.append(
            f"| `{gate.metric_name}` | {gate.baseline_value:.3f} | {gate.current_value:.3f} "
            f"| {-gate.percent_drop:+.1%} | {status} |"
        )

    lines += ["", f"_Allowed drop: {threshold:.0%}. Dataset: `{summary.dataset_version}`._"]
    if gate_result.dataset_changed:
        lines += [
            "",
            (
                "> ⚠️ The golden set changed since this baseline was recorded "
                f"(`{gate_result.baseline_dataset_version}` → "
                f"`{gate_result.run_dataset_version}`). "
                "Score moves may reflect the dataset, not the pipeline."
            ),
        ]
    return "\n".join(lines) + "\n"


def _print_scores(summary: RunSummary) -> None:
    table = Table(title=f"Run {summary.run_id} -- {summary.pipeline_name}@{summary.pipeline_version}")
    table.add_column("Metric")
    table.add_column("Mean Score", justify="right")
    for metric_name, value in summary.mean_scores.items():
        table.add_row(metric_name, f"{value:.3f}")
    console.print(table)


@app.command()
def run(
    pipeline: str = typer.Option("demo", help="Registered pipeline name to evaluate"),
    pipeline_version: str = typer.Option("dev", help="Version/git-sha tag for this pipeline build"),
    dataset: Path | None = typer.Option(  # noqa: B008 -- standard Typer pattern, ruff misflags Path-typed Options
        None, help="Golden-set JSONL path; defaults to the bundled set"
    ),
    use_judge: bool = typer.Option(True, help="Invoke the GPT-4o judge on failing scores"),
    gate: bool = typer.Option(
        False, help="Check the run against the baseline and exit non-zero on regression"
    ),
    baseline_file: Path | None = typer.Option(  # noqa: B008 -- standard Typer pattern, ruff misflags Path-typed Options
        None, help="Read the baseline from this JSON file instead of the database (used by CI)"
    ),
    summary_file: Path | None = typer.Option(  # noqa: B008 -- standard Typer pattern, ruff misflags Path-typed Options
        None, help="Write a markdown gate summary here (CI posts it as a PR comment)"
    ),
) -> None:
    """Run the golden dataset against a pipeline, score it, and persist the run."""
    dataset_path = dataset or DEFAULT_DATASET_PATH
    qa_pairs = load_golden_dataset(dataset_path)
    ds_version = dataset_version(dataset_path)
    pipeline_obj = _load_pipeline(pipeline)
    judge = GPT4oJudge() if use_judge else None

    summary = run_evaluation(
        pipeline_obj,
        qa_pairs,
        _default_metrics(),
        judge=judge,
        pipeline_name=pipeline,
        pipeline_version=pipeline_version,
        dataset_version=ds_version,
    )

    _print_scores(summary)

    if gate:
        gate_result = check_baseline_gate(summary, baseline_file=baseline_file)

        if gate_result.dataset_changed:
            console.print(
                "[yellow]Warning: the golden set changed since this baseline was recorded "
                f"({gate_result.baseline_dataset_version} -> {gate_result.run_dataset_version}). "
                "Score moves may reflect the dataset, not the pipeline.[/yellow]"
            )

        for metric_gate in gate_result.metrics:
            if metric_gate.baseline_value is None:
                console.print(f"  {metric_gate.metric_name}: no baseline set yet, skipping gate")
            else:
                status = "PASS" if metric_gate.passed else "FAIL"
                console.print(
                    f"  {metric_gate.metric_name}: {status} "
                    f"(baseline={metric_gate.baseline_value:.3f}, "
                    f"current={metric_gate.current_value:.3f}, "
                    f"drop={metric_gate.percent_drop:.1%})"
                )

        if summary_file is not None:
            summary_file.write_text(_gate_markdown(summary, gate_result), encoding="utf-8")

        if not gate_result.passed:
            console.print("[red]Gate FAILED: a metric regressed beyond the allowed threshold.[/red]")
            raise typer.Exit(code=1)
        console.print("[green]Gate passed.[/green]")


@app.command("baseline-export")
def baseline_export(
    run_id: int,
    output: Path = typer.Option(  # noqa: B008 -- standard Typer pattern, ruff misflags Path-typed Options
        Path("baseline.json"), help="Where to write the baseline snapshot"
    ),
) -> None:
    """Export a run's mean scores to a committed baseline file for CI to gate against."""
    summary = _summary_from_run(run_id)
    write_baseline_file(summary, output)
    console.print(f"Baseline written to {output} from run {run_id}:")
    for metric_name, value in summary.mean_scores.items():
        console.print(f"  {metric_name}: {value:.3f}")


@app.command("baseline-set")
def baseline_set(run_id: int) -> None:
    """Set the given run's mean scores as the new baseline for its pipeline."""
    summary = _summary_from_run(run_id)
    set_baseline_from_run(summary)
    console.print(f"Baseline set for {summary.pipeline_name!r} from run {run_id}:")
    for metric_name, value in summary.mean_scores.items():
        console.print(f"  {metric_name}: {value:.3f}")


@app.command()
def report(run_id: int) -> None:
    """Pretty-print a run's per-question pass/fail scores and any judge diagnostics."""
    with session_scope() as session:
        db_run = session.get(Run, run_id)
        if db_run is None:
            console.print(f"[red]No run with id {run_id}.[/red]")
            raise typer.Exit(code=1)
        results = session.scalars(select(Result).where(Result.run_id == run_id)).all()

        table = Table(title=f"Run {run_id} report")
        table.add_column("Question ID")
        table.add_column("Scores")
        table.add_column("Judge diagnosis")
        for result in results:
            scores_str = ", ".join(f"{k}={v:.2f}" for k, v in result.metric_scores.items())
            diagnosis_str = ""
            if result.judge_diagnosis:
                diagnosis_str = "; ".join(
                    f"[{d['metric']}] {d['verdict']}: {d['explanation']}"
                    for d in result.judge_diagnosis
                )
            table.add_row(result.question_id, scores_str, diagnosis_str)
        console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
