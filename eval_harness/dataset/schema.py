from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    DIRECT = "direct"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"
    ADVERSARIAL = "adversarial"
    EDGE_CASE = "edge_case"


class QAPair(BaseModel):
    """One entry in a golden dataset: a question plus the ground truth needed to score it."""

    id: str
    question: str
    ground_truth_answer: str
    ground_truth_contexts: list[str] = Field(default_factory=list)
    category: QuestionCategory = QuestionCategory.DIRECT
    difficulty: Literal["easy", "medium", "hard"] | None = None
