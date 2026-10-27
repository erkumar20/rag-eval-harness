#!/usr/bin/env bash
#
# Decides whether the eval gate's expensive steps need to run for this event, and writes
# `run=true|false` plus a `reason` to $GITHUB_OUTPUT.
#
# Why this exists: eval-gate is meant to be a *required* status check, and a required check
# that never triggers sits pending forever -- so a docs-only PR would be unmergeable. The
# workflow therefore triggers on every PR, and this script lets it finish green in seconds
# when nothing that can affect answer quality changed.
#
# Fails OPEN: if anything is uncertain it says run=true. Wrongly skipping would let a
# regression through on a green check, which is the failure this whole project exists to
# prevent; wrongly running only costs tokens.
#
# Inputs (env): EVENT_NAME, BASE_SHA. Output file defaults to stdout for local testing.

set -u

OUT="${GITHUB_OUTPUT:-/dev/stdout}"

emit() {
    echo "run=$1" >> "$OUT"
    echo "reason=$2" >> "$OUT"
    echo "eval scope: run=$1 ($2)"
}

# Anything that can change what the pipeline answers or how it is scored, plus the gate's own
# workflow and scripts (a change to those should be exercised by a real run, not assumed).
#
# baseline.json is deliberately NOT listed. The gate compares against the baseline on the base
# branch, so a PR that only edits baseline.json cannot change any score: evaluating it would
# just compare unchanged code to the old baseline, and could fail a legitimate re-baseline on
# run-to-run noise. Such a PR is instead reported loudly by baseline_changes.py. A PR that
# changes baseline.json *and* code still matches via the code paths, and is graded against the
# old baseline.
RELEVANT='^(rag_demo/|eval_harness/|\.github/workflows/eval-gate\.yml$|\.github/scripts/)'

if [ "${EVENT_NAME:-}" != "pull_request" ]; then
    emit true "manual or non-PR trigger"
    exit 0
fi

if [ -z "${BASE_SHA:-}" ]; then
    emit true "no base commit supplied, running to be safe"
    exit 0
fi

if ! changed="$(git diff --name-only "$BASE_SHA" HEAD 2>/dev/null)"; then
    emit true "could not diff against the base branch, running to be safe"
    exit 0
fi

if [ -z "$changed" ]; then
    emit true "diff came back empty, running to be safe"
    exit 0
fi

if echo "$changed" | grep -Eq "$RELEVANT"; then
    emit true "PR touches the pipeline, harness or baseline"
else
    emit false "PR touches nothing that can affect answer quality"
fi
