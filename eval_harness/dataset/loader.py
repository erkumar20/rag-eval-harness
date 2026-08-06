from __future__ import annotations

import hashlib
from pathlib import Path

from eval_harness.dataset.schema import QAPair

DEFAULT_DATASET_PATH = Path(__file__).parent / "golden_set.jsonl"


def load_golden_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[QAPair]:
    pairs: list[QAPair] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                pairs.append(QAPair.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid QAPair on line {line_no} of {path}: {exc}") from exc

    _check_unique_ids(pairs, path)
    return pairs


def _check_unique_ids(pairs: list[QAPair], path: Path) -> None:
    seen: set[str] = set()
    dupes: set[str] = set()
    for pair in pairs:
        (dupes if pair.id in seen else seen).add(pair.id)
    if dupes:
        raise ValueError(f"Duplicate QAPair ids in {path}: {sorted(dupes)}")


def dataset_version(path: Path = DEFAULT_DATASET_PATH) -> str:
    """Content hash of the dataset file, not a hand-maintained version string.

    Changes automatically the moment the golden set is edited, so baseline regression checks
    (Phase 9/10) can always tell "the dataset changed" apart from "the pipeline regressed"
    without anyone needing to remember to bump a version number by hand.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"golden_set-{digest[:10]}"
