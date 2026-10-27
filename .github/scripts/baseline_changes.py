#!/usr/bin/env python3
"""Reports how a pull request changes baseline.json, so lowering the bar is never silent.

The eval gate compares against the baseline on the *base branch*, so a PR cannot weaken its
own check. Changing the baseline therefore has to happen in a PR of its own -- and this script
makes that act loud: it writes a before/after table, with lowered metrics flagged, for the CI
job to post on the PR.

Usage: baseline_changes.py BASE_BASELINE PR_BASELINE OUTPUT_MD

Writes OUTPUT_MD only when something changed. Standard library only: it runs before any
dependency is installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def render(base: dict | None, pr: dict | None) -> str | None:
    base_metrics = (base or {}).get("metrics", {})
    pr_metrics = (pr or {}).get("metrics", {})
    base_dataset = (base or {}).get("dataset_version")
    pr_dataset = (pr or {}).get("dataset_version")

    if base_metrics == pr_metrics and base_dataset == pr_dataset:
        return None

    lines = [
        "## baseline.json changed in this PR",
        "",
        (
            "The eval gate compares against the baseline on the **base branch**, so this PR is "
            "not graded against its own numbers. Changing the baseline is deliberate and visible:"
        ),
        "",
        "| Metric | On base branch | In this PR | Change |",
        "| --- | --- | --- | --- |",
    ]
    lowered = False
    for name in sorted(set(base_metrics) | set(pr_metrics)):
        before, after = base_metrics.get(name), pr_metrics.get(name)
        if before is None:
            lines.append(f"| `{name}` | - | {after:.3f} | added |")
        elif after is None:
            lines.append(f"| `{name}` | {before:.3f} | - | removed |")
        elif before == after:
            lines.append(f"| `{name}` | {before:.3f} | {after:.3f} | unchanged |")
        else:
            change = (after - before) / before if before else 0.0
            flag = " **LOWERED**" if after < before else ""
            lowered = lowered or after < before
            lines.append(f"| `{name}` | {before:.3f} | {after:.3f} | {change:+.1%}{flag} |")

    if base_dataset != pr_dataset:
        lines += ["", f"Dataset version: `{base_dataset}` -> `{pr_dataset}`"]
    if lowered:
        lines += [
            "",
            (
                "> A lowered baseline makes later regressions easier to merge. Confirm the "
                "lower scores are an intended trade-off before approving."
            ),
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    base_path, pr_path, out_path = argv[1:]
    text = render(_load(base_path), _load(pr_path))
    if text is None:
        print("baseline.json is unchanged")
        return 0
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"baseline.json changed; wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
