"""Rebuild rag_demo's corpus from FastAPI's official docs (MIT-licensed, github.com/fastapi/fastapi).

Two-step fetch, because FastAPI's docs use a custom `{* path/to/file.py hl[...] *}` include
syntax (processed by their mkdocs build, not present in raw GitHub markdown) to inject code
examples. Fetching only the .md files leaves those as inert placeholder text -- most of the
actually-useful, answerable content (the code) would be missing. So this script also fetches
every referenced docs_src/*.py file and splices it back in as a real fenced code block.

Re-run this any time the corpus needs refreshing (see devlog/progress_log.md, Phase 3).
"""

from __future__ import annotations

import re
from pathlib import Path

import requests

REPO_RAW_BASE = "https://raw.githubusercontent.com/fastapi/fastapi/master"
CORPUS_DIR = Path(__file__).parent.parent / "rag_demo" / "data" / "corpus"

DOC_PAGES = [
    ("index.md", "index.md"),
    ("tutorial/first-steps.md", "tutorial_first-steps.md"),
    ("tutorial/path-params.md", "tutorial_path-params.md"),
    ("tutorial/query-params.md", "tutorial_query-params.md"),
    ("tutorial/query-params-str-validations.md", "tutorial_query-params-str-validations.md"),
    ("tutorial/path-params-numeric-validations.md", "tutorial_path-params-numeric-validations.md"),
    ("tutorial/body.md", "tutorial_body.md"),
    ("tutorial/body-multiple-params.md", "tutorial_body-multiple-params.md"),
    ("tutorial/body-fields.md", "tutorial_body-fields.md"),
    ("tutorial/body-nested-models.md", "tutorial_body-nested-models.md"),
    ("tutorial/cookie-params.md", "tutorial_cookie-params.md"),
    ("tutorial/header-params.md", "tutorial_header-params.md"),
    ("tutorial/response-model.md", "tutorial_response-model.md"),
    ("tutorial/response-status-code.md", "tutorial_response-status-code.md"),
    ("tutorial/request-forms.md", "tutorial_request-forms.md"),
    ("tutorial/request-files.md", "tutorial_request-files.md"),
    ("tutorial/request-forms-and-files.md", "tutorial_request-forms-and-files.md"),
    ("tutorial/handling-errors.md", "tutorial_handling-errors.md"),
    ("tutorial/path-operation-configuration.md", "tutorial_path-operation-configuration.md"),
    ("tutorial/background-tasks.md", "tutorial_background-tasks.md"),
    ("tutorial/middleware.md", "tutorial_middleware.md"),
    ("tutorial/cors.md", "tutorial_cors.md"),
    ("tutorial/sql-databases.md", "tutorial_sql-databases.md"),
    ("tutorial/testing.md", "tutorial_testing.md"),
    ("tutorial/bigger-applications.md", "tutorial_bigger-applications.md"),
    ("tutorial/dependencies/index.md", "tutorial_dependencies_index.md"),
    ("tutorial/dependencies/classes-as-dependencies.md", "tutorial_dependencies_classes-as-dependencies.md"),
    ("tutorial/security/first-steps.md", "tutorial_security_first-steps.md"),
    ("tutorial/security/oauth2-jwt.md", "tutorial_security_oauth2-jwt.md"),
]

INCLUDE_PATTERN = re.compile(r"\{\*\s*([^\s]+\.py)[^*]*\*\}")

_code_cache: dict[str, str] = {}


def fetch(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def fetch_code_snippet(docs_src_relative_path: str) -> str:
    """docs_src_relative_path looks like '../../docs_src/path_params/tutorial001_py310.py' --
    resolve it against the known repo-root docs_src/ directory rather than trying to
    replicate FastAPI's mkdocs path-resolution logic."""
    idx = docs_src_relative_path.find("docs_src/")
    repo_path = docs_src_relative_path[idx:]
    if repo_path not in _code_cache:
        _code_cache[repo_path] = fetch(f"{REPO_RAW_BASE}/{repo_path}").rstrip("\n")
    return _code_cache[repo_path]


def splice_code_includes(markdown: str) -> str:
    def replace(match: re.Match[str]) -> str:
        path = match.group(1)
        code = fetch_code_snippet(path)
        return f"```python\n{code}\n```"

    return INCLUDE_PATTERN.sub(replace, markdown)


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for src_path, dst_name in DOC_PAGES:
        markdown = fetch(f"{REPO_RAW_BASE}/docs/en/docs/{src_path}")
        markdown = splice_code_includes(markdown)
        (CORPUS_DIR / dst_name).write_text(markdown, encoding="utf-8")
        print(f"wrote {dst_name} ({len(markdown)} chars)")

    print(f"\nDone. {len(DOC_PAGES)} doc pages, {len(_code_cache)} unique code snippets spliced in.")


if __name__ == "__main__":
    main()
