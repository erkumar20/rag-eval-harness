# RAG Eval Harness

A modular evaluation framework for RAG pipelines: RAGAS metrics (faithfulness, context recall,
answer relevance), GPT-4o-as-judge structured failure diagnostics, a custom field-level
hallucination detector, and CI/CD regression gating via GitHub Actions.

> Status: under active development. See `docs/architecture.md` (coming in a later phase) for
> design details.

## Why

Most teams check whether a change to a RAG pipeline (new chunking, a swapped embedding model, a
prompt tweak) made things better or worse by eyeballing a handful of queries. This harness
automates that: run a versioned golden dataset through any RAG pipeline via a small adapter,
score it against faithfulness/context-recall/answer-relevance plus a custom hallucination check,
and block a PR automatically if any metric regresses past a configurable threshold.

## Project layout

```
rag_demo/        # small reference RAG pipeline used as the harness's test subject
eval_harness/    # the harness itself: interfaces, adapters, metrics, judge, storage, runner, CLI
tests/           # unit + integration tests
docs/            # architecture notes
```

## Local dev setup

No Docker required — Postgres runs natively via Homebrew and Qdrant runs in embedded
(no-server) mode.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

brew services start postgresql@16
createdb rag_eval   # once

cp .env.example .env   # fill in OPENAI_API_KEY / GROQ_API_KEY
```

CI (GitHub Actions) uses real Postgres + Qdrant service containers, so the environments only
diverge locally, not in CI.

## Quickstart

_Coming as the project is built out — see the plan for the full phase list._

## License

MIT
