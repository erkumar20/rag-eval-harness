# Corpus Source

29 pages from the official [FastAPI documentation](https://github.com/fastapi/fastapi)
(`docs/en/docs/`), fetched from the `master` branch. FastAPI is MIT-licensed; this is a small
subset used for a non-commercial evaluation/demo project, with attribution here.

The docs use a custom `{* docs_src/path/to/file.py hl[...] *}` include syntax (processed by
FastAPI's own mkdocs build, not present in raw GitHub markdown) to inject code examples. Raw
`.md` fetches leave these as inert placeholder text, which would gut most of the actually
answerable content. `scripts/build_corpus.py` fetches each referenced `docs_src/*.py` file
(119 of them) and splices the real code back in as fenced code blocks.

To refresh: `python scripts/build_corpus.py`, then `python rag_demo/ingest.py` to re-index.
