"""Tests for the helper scripts the eval-gate workflow depends on.

These guard the failure the whole project exists to prevent: a green check that evaluated
nothing. The scope script decides whether the evaluation runs at all, and the baseline script
is what makes a lowered baseline visible, so both are tested rather than trusted.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / ".github" / "scripts"


def _load_baseline_changes():
    spec = importlib.util.spec_from_file_location(
        "baseline_changes", SCRIPTS / "baseline_changes.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline_changes = _load_baseline_changes()


# --- eval_scope.sh -----------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    """A throwaway git repo with one base commit; returns (path, base sha)."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    for rel in ("rag_demo/x.py", "docs/a.md", "README.md", "baseline.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("base\n")
    git("add", "-A")
    git("commit", "-qm", "base")
    return tmp_path, git("rev-parse", "HEAD")


def _scope(repo_path: Path, base_sha: str, *, event: str = "pull_request") -> str:
    out = repo_path / "out.txt"
    out.write_text("")
    subprocess.run(
        ["bash", str(SCRIPTS / "eval_scope.sh")],
        cwd=repo_path,
        check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "EVENT_NAME": event,
             "BASE_SHA": base_sha, "GITHUB_OUTPUT": str(out)},
    )
    return next(
        line.split("=", 1)[1] for line in out.read_text().splitlines() if line.startswith("run=")
    )


def _commit_change(repo_path: Path, *files: str) -> None:
    for rel in files:
        path = repo_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.read_text() + "changed\n" if path.exists() else "new\n")
    subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-qm", "change"], cwd=repo_path, check=True)


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        (["docs/a.md"], "false"),
        (["README.md"], "false"),
        (["rag_demo_notes/x.md"], "false"),  # lookalike directory must not match
        (["tests/unit/test_x.py"], "false"),
        (["baseline.json"], "false"),  # baseline-only: cannot change a score
        ([".github/workflows/ci.yml"], "false"),
        (["rag_demo/x.py"], "true"),
        (["eval_harness/new.py"], "true"),
        ([".github/workflows/eval-gate.yml"], "true"),
        ([".github/scripts/eval_scope.sh"], "true"),
        ([".github/scripts/baseline_changes.py"], "true"),
        (["baseline.json", "rag_demo/x.py"], "true"),  # mixed PR is graded, against the old baseline
    ],
)
def test_scope_decides_from_the_changed_files(
    repo: tuple[Path, str], changed: list[str], expected: str
) -> None:
    repo_path, base = repo
    _commit_change(repo_path, *changed)
    assert _scope(repo_path, base) == expected


def test_scope_fails_open_when_it_cannot_tell(repo: tuple[Path, str]) -> None:
    """Wrongly skipping lets a regression through on a green check; wrongly running only
    costs tokens. Every uncertain case must therefore say run=true."""
    repo_path, base = repo
    assert _scope(repo_path, base, event="workflow_dispatch") == "true"  # manual run
    assert _scope(repo_path, "") == "true"  # no base commit supplied
    assert _scope(repo_path, "deadbeef" * 5) == "true"  # base commit does not exist
    assert _scope(repo_path, base) == "true"  # empty diff


# --- baseline_changes.py -----------------------------------------------------------------


def _baseline(**metrics: float) -> dict:
    return {"dataset_version": "golden_set-aaa", "metrics": metrics}


def test_unchanged_baseline_reports_nothing() -> None:
    same = _baseline(faithfulness=0.9, context_recall=0.8)
    assert baseline_changes.render(same, dict(same)) is None


def test_a_lowered_metric_is_flagged_loudly() -> None:
    text = baseline_changes.render(_baseline(faithfulness=0.9), _baseline(faithfulness=0.5))
    assert text is not None
    assert "LOWERED" in text
    assert "-44.4%" in text
    assert "lowered baseline" in text.lower()


def test_a_raised_metric_is_reported_but_not_flagged() -> None:
    text = baseline_changes.render(_baseline(faithfulness=0.5), _baseline(faithfulness=0.9))
    assert text is not None
    assert "+80.0%" in text
    assert "LOWERED" not in text


def test_added_and_removed_metrics_are_reported() -> None:
    text = baseline_changes.render(_baseline(old=0.5), _baseline(new=0.6))
    assert text is not None
    assert "added" in text and "removed" in text


def test_a_dataset_version_change_is_reported_even_if_scores_match() -> None:
    base = _baseline(faithfulness=0.9)
    pr = {**base, "dataset_version": "golden_set-bbb"}
    text = baseline_changes.render(base, pr)
    assert text is not None
    assert "golden_set-aaa" in text and "golden_set-bbb" in text


def test_cli_writes_the_report_only_when_something_changed(tmp_path: Path) -> None:
    base, same, lower, out = (tmp_path / n for n in ("base.json", "same.json", "lower.json", "out.md"))
    base.write_text(json.dumps(_baseline(faithfulness=0.9)))
    same.write_text(json.dumps(_baseline(faithfulness=0.9)))
    lower.write_text(json.dumps(_baseline(faithfulness=0.6)))

    assert baseline_changes.main(["x", str(base), str(same), str(out)]) == 0
    assert not out.exists()

    assert baseline_changes.main(["x", str(base), str(lower), str(out)]) == 0
    assert "LOWERED" in out.read_text()
