#!/usr/bin/env bash
#
# Terminal demo: show that a quality regression invisible to tests is caught by the eval gate.
#
#   ./scripts/demo.sh
#
# Uses a 5-question subset of the golden set (a full 35-question run is ~15 minutes and ~87k
# generator tokens, against a 200k/day free tier) and builds its own baseline from the clean
# run, so the comparison is like-for-like.
#
# Costs roughly 12k generator tokens total. Restores any file it edits, including on Ctrl-C.

set -uo pipefail
cd "$(dirname "$0")/.."

RETRIEVER="rag_demo/retriever.py"
BACKUP="$(mktemp)"
SUBSET="$(mktemp)"
BASELINE="$(mktemp)"

cleanup() {
    # The regression is a real edit to tracked source -- restore it whatever happens.
    [ -f "$BACKUP" ] && cp "$BACKUP" "$RETRIEVER"
    rm -f "$BACKUP" "$SUBSET" "$BASELINE"
}
trap cleanup EXIT INT TERM

cp "$RETRIEVER" "$BACKUP"

say()  { printf '\n\033[1;36m%s\033[0m\n' "$*"; }
note() { printf '\033[0;90m%s\033[0m\n' "$*"; }
pause() { sleep "${DEMO_PAUSE:-2}"; }

# ---------------------------------------------------------------------------
say "1/5  The golden dataset"
note "Hand-written questions over the FastAPI docs, each with a known-good answer."
head -5 eval_harness/dataset/golden_set.jsonl > "$SUBSET"
python3 -c "
import json, sys
for line in open('$SUBSET'):
    qa = json.loads(line)
    print(f\"  {qa['id']}  [{qa['category']}]  {qa['question'][:62]}\")
"
pause

# ---------------------------------------------------------------------------
say "2/5  Score the pipeline as it stands, and record that as the baseline"
note "Each line is a real question scored against live models."
evalharness run \
    --pipeline demo \
    --pipeline-version demo-clean \
    --dataset "$SUBSET" \
    --no-use-judge \
  2>&1 | grep -Ev 'Loading weights|HF_TOKEN|^$'

# PIPESTATUS[0], not $?: the latter would report grep's status, not the harness's.
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    printf '\n\033[0;31mBaseline run failed -- aborting demo.\033[0m\n'
    exit 1
fi

CLEAN_RUN=$(python3 -c "
from sqlalchemy import select
from eval_harness.storage.db import session_scope
from eval_harness.storage.models import Run
with session_scope() as s:
    print(s.scalars(select(Run.id).order_by(Run.id.desc()).limit(1)).one())
")
evalharness baseline-export "$CLEAN_RUN" --output "$BASELINE"
pause

# ---------------------------------------------------------------------------
say "3/5  Introduce a regression -- one character"
note "Flip the retrieval ranking so it returns its WORST matches instead of its best."
python3 - "$RETRIEVER" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
assert "reverse=True)[:k]" in src, "demo expects the descending RRF sort to be present"
p.write_text(src.replace("reverse=True)[:k]", "reverse=False)[:k]"))
PY
git --no-pager diff --unified=0 -- "$RETRIEVER" | grep -E '^[+-] ' | grep -v '^[+-][+-]'
pause

# ---------------------------------------------------------------------------
say "4/5  Nothing is broken, as far as any normal check can tell"
note "Linting, type checking and the full test suite -- all green on the regressed code."
ruff check "$RETRIEVER" && mypy "$RETRIEVER" 2>&1 | tail -1
pytest -q 2>&1 | tail -1
note "This is the point: no exception, no failing test. Only answer quality moved."
pause

# ---------------------------------------------------------------------------
say "5/5  The eval gate catches it"
evalharness run \
    --pipeline demo \
    --pipeline-version demo-regressed \
    --dataset "$SUBSET" \
    --no-use-judge \
    --gate \
    --baseline-file "$BASELINE" \
  2>&1 | grep -Ev 'Loading weights|HF_TOKEN|^$'
GATE_EXIT=${PIPESTATUS[0]}

say "Gate exit code: $GATE_EXIT"
note "Non-zero fails the GitHub Actions check, which blocks the merge."
