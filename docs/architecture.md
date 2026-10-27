# Architecture

How the harness is put together and why. Covers the adapter boundary, the metric pipeline, the
judge, the storage schema and the gating logic — including the decisions that were reversed
after measurement, which are usually the interesting part.

## Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ cli.py            run · report · baseline-set · baseline-export │
├─────────────────────────────────────────────────────────────────┤
│ runner/                                                         │
│   evaluate.py     loop questions → score → judge → persist      │
│   baseline.py     compare to baseline, compute drop, pass/fail  │
├──────────────┬──────────────┬───────────────┬───────────────────┤
│ metrics/     │ judge/       │ dataset/      │ storage/          │
│ RAGAS +      │ GPT-4o →     │ QAPair +      │ SQLAlchemy +      │
│ hallucination│ Diagnosis    │ content hash  │ Alembic → Postgres│
├──────────────┴──────────────┴───────────────┴───────────────────┤
│ interfaces/pipeline.py   RAGPipelineInterface (Protocol)        │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                    any pipeline satisfying query()
                              │
                      rag_demo/ (reference subject)
```

## The adapter boundary

The entire contract between the harness and the thing being evaluated:

```python
class PipelineOutput(TypedDict):
    answer: str
    contexts: list[str]

@runtime_checkable
class RAGPipelineInterface(Protocol):
    def query(self, question: str) -> PipelineOutput: ...
```

It is a `Protocol`, so conformance is **structural, not nominal** — a pipeline does not inherit
from anything or import anything from this package. It just needs a `query` method of that
shape.

This matters for two reasons:

1. **Adapters stay tiny.** Wrapping a LangChain chain, a REST endpoint or an in-process
   function is a handful of lines that map its output into `{"answer", "contexts"}`.
2. **It makes a public/private split free.** The same harness can be `pip install`ed into a
   private project and pointed at a proprietary pipeline via a private adapter, with no client
   data or internals entering this repo.

`rag_demo/` is simply the first implementation of that contract — hybrid retrieval (vector +
BM25, merged with reciprocal rank fusion) over 30 FastAPI documentation pages, answered by a
Groq-hosted model.

## Dataset

`QAPair` holds a question, a ground-truth answer, optional ground-truth contexts, and a
category: `direct` (14), `multi_hop` (6), `adversarial` (7), `unanswerable` (5), `edge_case` (3).
The `unanswerable` set matters most — it is how you detect a pipeline that hallucinates rather
than declining.

The dataset version is a **content hash of the file**, not a hand-maintained string:

```python
f"golden_set-{sha256(path.read_bytes()).hexdigest()[:10]}"
```

Nobody has to remember to bump it. Every run records the hash, and the gate warns when a
baseline was recorded against a different dataset — so "the dataset changed" is never silently
misread as "the pipeline regressed."

## Metrics

All metrics implement one interface:

```python
class Metric(ABC):
    name: str
    def score(self, eval_input: EvalInput) -> MetricResult: ...
```

**RAGAS-backed** (`metrics/ragas_metrics.py`): faithfulness, context recall, answer relevance.
Each wraps a RAGAS metric and delegates its internal LLM work to `gpt-4o-mini` for cost.

Two non-obvious details, both the result of debugging rather than design:

- **`max_tokens=4096`, explicitly.** RAGAS's `llm_factory` defaults to 1024, which silently
  truncates structured verdict output mid-JSON on longer answers. It surfaced as an
  `IncompleteOutputException` ten questions into a real run, not in any smaller test.
- **The OpenAI client and each RAGAS metric object are built lazily, per thread.** They were
  originally constructed once in `__init__`. Because each `score()` call runs
  `asyncio.run(...)`, a shared async client ends up driven from several different event loops,
  whose internal locks and pools are loop-bound. Nothing raises — it just degrades badly.

**Questions the corpus cannot answer are scored differently.** A correct refusal is the right
behaviour on an out-of-scope question, but RAGAS `answer_relevance` guesses the question back from
the answer alone and forces the score to 0 when the answer reads as evasive. So:

- `answer_relevance` raises `ValueError` ("not applicable") for `unanswerable` questions, and the
  runner skips that metric for them. Its mean is over answerable questions only.
- `refusal_correctness` (`metrics/refusal.py`) scores only those questions: 1.0 if the answer
  declines *without* supplying an outside-knowledge answer, 0.0 if it answers anyway. Its mean is
  the fraction of out-of-scope questions correctly declined, which is the hallucination check for
  out-of-scope questions. It uses a small LLM classifier (a string match on "sorry" would be
  brittle); the classifier is injectable so tests need no API key.
- `context_recall` skips them too. Its reference for these questions is "not answerable from this
  documentation", and whether retrieved text "supports" that is close to a coin flip: re-scoring
  *identical* stored inputs five times gave `[1,0,1,1,1]` and `[0,0,0,0,1]` on two of the five
  out-of-scope questions, while five answerable controls were identical every time.
- Reports show per-category means, because one overall mean hides which kind of question moved.

Re-scoring the first full run this way: `answer_relevance` 0.558 over all 35 becomes 0.628 over
the 30 answerable questions, and `refusal_correctness` is 1.000 (all 5 out-of-scope questions
declined). `answer_relevance` is also deterministic given the answer (its three generated
questions are identical), so its run-to-run movement comes only from answer changes.

**Custom hallucination metric** (`metrics/hallucination.py`): operates on *structured* answers,
checking each field independently against the specific contexts it cites. Two failure modes per
field — a fabricated citation (a `cited_chunk_id` that was never retrieved) and an ungrounded
value (not present in the context it cites) — reported as a per-field pass/fail rather than one
aggregate number.

It uses **strict verbatim matching and no semantic fallback**, which was a reversal. An earlier
version compared embeddings for values that didn't match literally. Calibration killed it: real
paraphrases scored 0.73–0.95 similarity, while same-topic claims with a swapped number or name
— the canonical hallucination — scored 0.79–0.81. The fabrications sometimes scored *higher*
than legitimate paraphrases, so no threshold could separate them. Embedding similarity is a
topic signal; it is close to blind to a single swapped digit, which is exactly what this metric
exists to catch. For structured field extraction, near-verbatim grounding is also the correct
expectation — paraphrase tolerance is what RAGAS faithfulness is for.

## Judge

```python
class JudgeLLM(ABC):
    def judge(self, question, contexts, answer, metric_name, score) -> Diagnosis: ...
```

`Diagnosis` is a Pydantic model (`metric`, `verdict`, `explanation`, `evidence_span`) produced
via `instructor`, so the output is guaranteed parseable — no hand-rolled JSON handling or retry
logic.

Two deliberate choices:

- **Cost control:** the judge only runs on metrics scoring below `judge_score_threshold`
  (default 0.7). Passing answers are never sent. `answer_relevance` is excluded
  (`judge_excluded_metrics`): its low scores come from how it works, not from bad answers, and
  24 of the 25 judge calls on the first run were spent contradicting it.
- **It may disagree.** The prompt explicitly tells the judge that if a low score looks wrong on
  inspection, it should say so. This is what makes it diagnostic rather than a rubber stamp —
  on the first full run it returned `pass` on 23 of 24 low `answer_relevance` scores, which
  prompted the investigation that found why (above).

The gate's verdict never consults judge output; the judge explains, it does not decide. That
separation is why CI can run with `--no-use-judge` and still gate correctly.

## Storage

Four tables (`storage/models.py`, migrated with Alembic):

| Table | Purpose |
|---|---|
| `pipeline_versions` | name + version/git SHA — keeps runs from different builds comparable |
| `runs` | one execution: pipeline version, dataset version, started/finished |
| `results` | one row per question: answer, contexts, `metric_scores` and `judge_diagnosis` as JSONB |
| `baselines` | current reference value per (pipeline, metric), unique-constrained |

Two details worth noting:

- **Results are committed as they are scored**, not batched at the end. A run costs real money
  per question; persisting only on completion meant a timeout, cancellation or crash discarded
  everything. A run with results but no `finished_at` is identifiable as partial — which also
  guards against reading means off an incomplete run and mistaking them for a result.
- **`judge_diagnosis` uses `JSONB(none_as_null=True)`.** SQLAlchemy otherwise stores Python
  `None` as JSON `null`, which is not SQL `NULL`, so `WHERE judge_diagnosis IS NOT NULL` matches
  every row — and reports that the judge ran on all 35 questions when it ran on 25.

## Gating

```python
percent_drop = (baseline_value - current_value) / baseline_value
passed = percent_drop <= threshold      # default 0.05
```

Positive drop means the score fell; an improvement is negative and always passes. A metric with
no baseline passes by default, so a pipeline's first run cannot fail its own gate.

**Baselines are committed to the repo** (`baseline.json`), not read from the database, because
CI gets a fresh empty Postgres on every job -- a database-only baseline would always read back as
"no baseline yet" and the gate would silently pass every time. A green check that can never fail
is worse than no check.

**A PR is graded against the baseline on the base branch, never its own copy.** Committing the
file alone was not enough: a PR could lower `baseline.json` and then pass against its own
weakened bar. The workflow therefore extracts `baseline.json` from the PR's base commit
(`git show $BASE_SHA:baseline.json`) and passes that to the gate. If the base branch has no
baseline, the job fails rather than passing by default.

This also means changing the baseline is a deliberate, separate act:

- A PR that **only** edits `baseline.json` cannot change any score, so it skips the evaluation
  (running it would compare unchanged code to the old baseline and could fail a legitimate
  re-baseline on run-to-run noise). Instead the PR gets a comment with a before/after table, with
  lowered metrics flagged `LOWERED`, so the change cannot go unnoticed.
- A PR that changes the baseline **and** code is evaluated against the old baseline. It cannot
  lower the bar and pass in the same change.

On a single-maintainer repo there is no second reviewer, so lowering the baseline is still
possible by opening that dedicated PR and merging it yourself. What the design guarantees is that
it takes an explicit, visible, separate step; it does not pretend to prevent a maintainer from
approving their own change.

### Known weakness

The threshold is a single 5% figure for all metrics. Two CI runs of functionally identical code
scored `context_recall` about 10% above the baseline (0.905 and 0.914 vs 0.829), which would fail
a gate at 5% if it were a drop.

Much of that swing has now been traced: the five out-of-scope questions are 1/35 of the mean each
(0.029), their `context_recall` flips between identical runs, and three flips is 0.086 — the size
of the gap. Those questions no longer contribute to `context_recall` (see Metrics), which should
remove most of the noise. Two things remain open: the committed baseline predates that change and
must be regenerated, and the remaining variance on answerable questions has not yet been measured
across repeated full runs, so per-metric thresholds are still to be derived from data. Measuring
it is constrained by the token budget below.

## Operational constraints

The demo generator is a reasoning model and spends ~2.5k tokens per question including hidden
chain-of-thought, so a 35-question run is ~87k tokens against a 200k/day free tier — roughly
**two full runs per day**, CI and local combined.

Concurrency is therefore **1 by default**. Raising it is counterproductive on a rate-limited
tier: parallel requests hit per-minute limits, both SDKs auto-retry with backoff, and the
resulting thrash is slower than running sequentially. A five-worker run of the golden set had
not finished after 24 minutes; the sequential equivalent takes ~15.
