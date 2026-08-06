from __future__ import annotations

from pathlib import Path

import pytest

from eval_harness.dataset.loader import dataset_version, load_golden_dataset

VALID_LINE = (
    '{"id": "q001", "question": "Q?", "ground_truth_answer": "A.", '
    '"ground_truth_contexts": [], "category": "direct", "difficulty": "easy"}'
)


def write_dataset(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_loads_real_golden_dataset():
    pairs = load_golden_dataset()
    assert len(pairs) == 35
    ids = [p.id for p in pairs]
    assert len(ids) == len(set(ids))  # no duplicates in the actual shipped dataset


def test_skips_blank_lines(tmp_path: Path):
    path = write_dataset(tmp_path, [VALID_LINE, "", "   ", VALID_LINE.replace("q001", "q002")])
    pairs = load_golden_dataset(path)
    assert len(pairs) == 2


def test_duplicate_ids_raise(tmp_path: Path):
    path = write_dataset(tmp_path, [VALID_LINE, VALID_LINE])
    with pytest.raises(ValueError, match="Duplicate QAPair ids"):
        load_golden_dataset(path)


def test_invalid_line_raises_with_line_number(tmp_path: Path):
    path = write_dataset(tmp_path, [VALID_LINE, "{not valid json"])
    with pytest.raises(ValueError, match="line 2"):
        load_golden_dataset(path)


def test_dataset_version_changes_with_content(tmp_path: Path):
    path = write_dataset(tmp_path, [VALID_LINE])
    v1 = dataset_version(path)

    path.write_text(VALID_LINE.replace("Q?", "Different question?") + "\n", encoding="utf-8")
    v2 = dataset_version(path)

    assert v1 != v2
    assert v1.startswith("golden_set-")


def test_dataset_version_stable_for_unchanged_file(tmp_path: Path):
    path = write_dataset(tmp_path, [VALID_LINE])
    assert dataset_version(path) == dataset_version(path)
