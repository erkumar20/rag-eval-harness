from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from eval_harness.config import get_settings
from eval_harness.dataset.schema import QAPair
from eval_harness.interfaces.pipeline import RAGPipelineInterface
from eval_harness.judge.base import JudgeLLM
from eval_harness.judge.schema import Diagnosis
from eval_harness.metrics.base import Metric
from eval_harness.schemas import EvalInput, EvalResult, MetricResult
from eval_harness.storage.db import session_scope
from eval_harness.storage.repository import save_run


@dataclass
class RunSummary:
    """What `run_evaluation` hands back: the persisted run's id plus enough to print a
    report or feed straight into the baseline gate (Phase 10) without re-querying storage."""

    run_id: int
    pipeline_name: str
    pipeline_version: str
    dataset_version: str | None
    mean_scores: dict[str, float] = field(default_factory=dict)
    results: list[EvalResult] = field(default_factory=list)


def _score_question(
    qa: QAPair,
    pipeline: RAGPipelineInterface,
    metrics: list[Metric],
    judge: JudgeLLM | None,
    judge_threshold: float,
    pipeline_version: str,
    dataset_version: str | None,
) -> EvalResult:
    output = pipeline.query(qa.question)
    eval_input = EvalInput(
        question=qa.question,
        answer=output["answer"],
        contexts=output["contexts"],
        ground_truth_answer=qa.ground_truth_answer,
        ground_truth_contexts=qa.ground_truth_contexts,
    )

    metric_results: list[MetricResult] = []
    diagnoses: list[Diagnosis] = []
    for metric in metrics:
        try:
            result = metric.score(eval_input)
        except ValueError:
            # Not every metric applies to every pipeline -- e.g. HallucinationMetric
            # (Phase 7) requires EvalInput.structured_answer, which the demo pipeline never
            # populates. Skip rather than fail the whole run over an inapplicable metric.
            continue
        metric_results.append(result)

        if judge is not None and result.score < judge_threshold:
            diagnoses.append(
                judge.judge(
                    question=qa.question,
                    contexts=eval_input.contexts,
                    answer=eval_input.answer,
                    metric_name=result.metric_name,
                    score=result.score,
                )
            )

    return EvalResult(
        qa_id=qa.id,
        question=qa.question,
        answer=eval_input.answer,
        contexts=eval_input.contexts,
        pipeline_version=pipeline_version,
        dataset_version=dataset_version,
        metric_results=metric_results,
        judge_diagnoses=diagnoses,
    )


def _mean_scores(results: list[EvalResult]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for eval_result in results:
        for metric_result in eval_result.metric_results:
            sums[metric_result.metric_name] = (
                sums.get(metric_result.metric_name, 0.0) + metric_result.score
            )
            counts[metric_result.metric_name] = counts.get(metric_result.metric_name, 0) + 1
    return {name: sums[name] / counts[name] for name in sums}


def run_evaluation(
    pipeline: RAGPipelineInterface,
    dataset: list[QAPair],
    metrics: list[Metric],
    *,
    pipeline_name: str,
    pipeline_version: str,
    judge: JudgeLLM | None = None,
    dataset_version: str | None = None,
    max_workers: int = 1,
) -> RunSummary:
    """Scores every question in `dataset` against `pipeline` with every metric in `metrics`,
    invoking `judge` only on scores below the configured threshold (cost control from
    Phase 6), persists the full run via Phase 8's storage layer, and returns a summary.

    Parallelism across questions is supported but defaults to 1 (sequential), because the
    free API tiers this project targets are rate-limited well below what concurrency needs:
    Groq's free tier allows 8k tokens/minute, and one reasoning-model answer costs ~2.5k, so
    even 5 concurrent questions 429 immediately. The SDKs then auto-retry with backoff, which
    turns into a retry-thrash loop that is *slower* than running sequentially (measured: a
    5-worker run of the golden set hadn't finished after 24 minutes, versus ~25s/question
    sequentially). Raise this only against API tiers with headroom to match.
    """
    judge_threshold = get_settings().judge_score_threshold

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(
            pool.map(
                lambda qa: _score_question(
                    qa, pipeline, metrics, judge, judge_threshold, pipeline_version, dataset_version
                ),
                dataset,
            )
        )

    with session_scope() as session:
        run = save_run(
            session,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            dataset_version=dataset_version,
            results=results,
        )
        run_id = run.id

    return RunSummary(
        run_id=run_id,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        dataset_version=dataset_version,
        mean_scores=_mean_scores(results),
        results=results,
    )
