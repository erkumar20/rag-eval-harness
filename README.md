# RAG Eval Harness

Quality regression testing for RAG pipelines. Scores a versioned golden dataset against any
pipeline, explains failures with an LLM judge, and **blocks a pull request when answer quality
drops** — the same way CI blocks a failing test.

[![CI](https://github.com/erkumar20/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/erkumar20/rag-eval-harness/actions/workflows/ci.yml)

## The problem

RAG pipelines degrade silently. Change a chunk size, swap an embedding model, tweak a prompt —
nothing crashes, every test still passes, and the code is "correct." Answer quality just quietly
gets worse, and you find out from users weeks later.

Conventional testing can't catch this, because there is no exception to throw. The system works
exactly as written; it just works *worse*. So quality has to be measured, given a baseline, and
enforced.

## What it does

```
golden dataset ──► pipeline ──► metrics ──► judge (on failures) ──► Postgres
                                   │
                                   └──► baseline comparison ──► CI gate (pass/fail)
```

1. **Golden dataset** — 35 hand-written questions over the FastAPI docs, each with a known-good
   answer. Content-hashed, so changing it is detectable and never silently confused with a
   pipeline change.
2. **Metrics** — RAGAS faithfulness, context recall and answer relevance, plus a custom
   field-level hallucination detector for structured output.
3. **Judge** — GPT-4o inspects any metric that scores below threshold and returns a structured
   diagnosis. It is explicitly allowed to *disagree* with the metric (see
   [What the judge found](#what-the-judge-found)).
4. **Storage** — every run, result and judge diagnosis persisted to Postgres, with results
   committed as they are scored so an interrupted run keeps what it already paid for.
5. **Gate** — compares a run's mean scores to a committed baseline and fails the build when any
   metric drops beyond the threshold (default 5%).

## Results on the reference pipeline

Baseline from a real run over all 35 questions:

| Metric | Score |
|---|---|
| faithfulness | 0.929 |
| context_recall | 0.829 |
| answer_relevance | 0.558 |

The gate was then validated end to end on a real pull request: a deliberate retrieval
regression was introduced, CI **blocked the merge**, and the same PR went green once reverted.

<!-- Gate comment posted automatically on the PR -->
```
## RAG eval gate — demo

❌ Gate FAILED

| Metric           | Baseline | Current | Change | Status |
| ---------------- | -------- | ------- | ------ | ------ |
| faithfulness     | 0.929    | 0.869   | -6.4%  | FAIL   |
| context_recall   | 0.829    | 0.771   | -6.9%  | FAIL   |
| answer_relevance | 0.558    | 0.452   | -19.0% | FAIL   |

Allowed drop: 5%. Dataset: golden_set-d01d724df8.
```

## What the judge found

The judge is not decoration. On the first full run it disagreed with `answer_relevance`
**23 out of 24 times** — the metric scored answers poorly that GPT-4o, reading them, judged
perfectly relevant. By contrast `context_recall`'s failures were mostly judged genuine
(5 fail / 1 pass), and traced to a real retrieval gap.

That asymmetry is the argument for having a judge layer at all: a single aggregate score tells
you something moved, not whether it was the pipeline or the metric that was wrong.

It also led to the cause. RAGAS's `answer_relevance` never sees the question: it guesses the
question back from the answer and compares the guess to the real one. So a correct code-only
answer scores 0.22 ("What does the provided Python code do?" vs the real question), and a correct
**refusal** is forced to 0 because it reads as evasive. Questions the corpus cannot answer are
therefore scored by a dedicated `refusal_correctness` check instead, and `answer_relevance` is
averaged over answerable questions only. It is still kept as a regression signal: it fell 54%
when retrieval was deliberately broken.

## Design

**Pipelines plug in structurally, not by inheritance.** The only contract is:

```python
class RAGPipelineInterface(Protocol):
    def query(self, question: str) -> PipelineOutput: ...   # {"answer": str, "contexts": list[str]}
```

Any pipeline satisfying that shape — a LangChain chain, a REST call, an in-process function —
can be evaluated by writing a small adapter, with no dependency on this package's class
hierarchy. The bundled `rag_demo` pipeline (hybrid vector + BM25 retrieval over 30 FastAPI doc
pages) is just the first such test subject.

See [docs/architecture.md](docs/architecture.md) for the metric pipeline, judge design, storage
schema and gating logic.

## Quickstart

Requires Python 3.11+, Postgres, and API keys for OpenAI (metrics + judge) and Groq (the demo
pipeline's generator). No Docker needed — Qdrant runs embedded, Postgres natively.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

brew services start postgresql@16
createdb rag_eval

cp .env.example .env     # add OPENAI_API_KEY and GROQ_API_KEY
alembic upgrade head     # create tables
python -m rag_demo.ingest   # chunk + embed the corpus into Qdrant
```

Then:

```bash
# Score the golden set and persist the run
evalharness run --pipeline demo --pipeline-version v1

# Per-question scores and judge diagnostics
evalharness report <run-id>

# Record a run as the baseline to gate against
evalharness baseline-export <run-id> --output baseline.json

# Score and fail the build on regression (how CI invokes it)
evalharness run --pipeline demo --no-use-judge --gate --baseline-file baseline.json
```

A full 35-question run takes ~15 minutes and is sequential by default — see
[Cost and rate limits](#cost-and-rate-limits).

## CI

Two workflows, deliberately separate:

| Workflow | Trigger | Needs API keys | Runtime |
|---|---|---|---|
| `ci.yml` | every push and PR | no | ~3 min |
| `eval-gate.yml` | PRs touching `rag_demo/` or `eval_harness/` | yes | ~15 min |

`ci.yml` runs ruff, mypy and the full test suite with **no secrets set** — every test fakes the
pipeline, metrics and judge, so a test that starts needing a live key fails the build loudly.

`eval-gate.yml` runs the real evaluation, posts the metric-delta table as a PR comment, and
exits non-zero on regression. It is scoped narrowly because it costs real tokens.

The gate grades a PR against the baseline on the **base branch**, not the PR's own copy, so a PR
cannot lower the bar for itself. Changing `baseline.json` is its own PR, and the bot flags any
lowered metric. `main` is protected: both checks must pass before a merge.

## Cost and rate limits

A full run is ~87k generator tokens. On Groq's free tier (200k tokens/day) that allows roughly
**two full runs per day** across CI and local combined, which is why the gate is scoped to
pipeline-touching PRs and why concurrency defaults to 1 — parallel requests hit per-minute
limits and the resulting retry-backoff is *slower* than running sequentially.

## Known limitations

Measured, not hidden:

- **The 5% threshold is tighter than metric noise.** Two runs of identical code produced a
  +9.2% swing in `context_recall`, so a clean PR can fail the gate by chance. Fixing this
  properly needs per-metric thresholds derived from measured variance.
- **`answer_relevance` is a change detector, not an absolute score.** Even on answerable
  questions it rewards answers whose wording lets the question be reconstructed, so correct
  code-only, yes/no and long multi-part answers score low. Read its movement, not its level.
- **The committed baseline predates the refusal check**, so `refusal_correctness` is reported
  but not yet gated until the baseline is regenerated.
- The hallucination metric requires structured output, so it is opt-in and does not run against
  the free-text demo pipeline.

## License

MIT
