"""Phase 5 step 5: run all three RAGAS metrics across the full golden dataset, using the
real DemoRAGPipeline's actual outputs (not manufactured examples). Prints per-question scores
and per-category averages.

This is a verification/demo script, not the real orchestration layer -- Phase 9's runner will
handle persistence, the judge, and baseline gating. This just proves the metrics work
end-to-end against the real pipeline before building more on top of them.
"""

from __future__ import annotations

from collections import defaultdict

from eval_harness.dataset.loader import load_golden_dataset
from eval_harness.metrics.ragas_metrics import (
    AnswerRelevanceMetric,
    ContextRecallMetric,
    FaithfulnessMetric,
)
from eval_harness.schemas import EvalInput
from rag_demo.pipeline import DemoRAGPipeline


def main() -> None:
    pairs = load_golden_dataset()
    pipeline = DemoRAGPipeline()

    faithfulness = FaithfulnessMetric()
    context_recall = ContextRecallMetric()
    answer_relevance = AnswerRelevanceMetric()

    per_category_scores: dict[str, list[dict[str, float]]] = defaultdict(list)

    for pair in pairs:
        result = pipeline.query(pair.question)
        eval_input = EvalInput(
            question=pair.question,
            answer=result["answer"],
            contexts=result["contexts"],
            ground_truth_answer=pair.ground_truth_answer,
            ground_truth_contexts=pair.ground_truth_contexts,
        )

        f = faithfulness.score(eval_input)
        cr = context_recall.score(eval_input)
        ar = answer_relevance.score(eval_input)

        scores = {"faithfulness": f.score, "context_recall": cr.score, "answer_relevance": ar.score}
        per_category_scores[pair.category.value].append(scores)

        print(
            f"{pair.id} [{pair.category.value}] "
            f"F={f.score:.2f} CR={cr.score:.2f} AR={ar.score:.2f}  {pair.question[:60]}"
        )

    print("\n--- Averages by category ---")
    for category, rows in per_category_scores.items():
        n = len(rows)
        avg_f = sum(r["faithfulness"] for r in rows) / n
        avg_cr = sum(r["context_recall"] for r in rows) / n
        avg_ar = sum(r["answer_relevance"] for r in rows) / n
        print(f"{category:12s} (n={n:2d})  F={avg_f:.3f}  CR={avg_cr:.3f}  AR={avg_ar:.3f}")

    all_rows = [row for rows in per_category_scores.values() for row in rows]
    n = len(all_rows)
    print(
        f"\n{'OVERALL':12s} (n={n:2d})  "
        f"F={sum(r['faithfulness'] for r in all_rows) / n:.3f}  "
        f"CR={sum(r['context_recall'] for r in all_rows) / n:.3f}  "
        f"AR={sum(r['answer_relevance'] for r in all_rows) / n:.3f}"
    )


if __name__ == "__main__":
    main()
