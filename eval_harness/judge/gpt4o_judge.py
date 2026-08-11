from __future__ import annotations

import instructor
from openai import OpenAI

from eval_harness.config import get_settings
from eval_harness.judge.base import JudgeLLM
from eval_harness.judge.prompts import SYSTEM_PROMPT, build_judge_prompt
from eval_harness.judge.schema import Diagnosis


class GPT4oJudge(JudgeLLM):
    """LLM-as-judge: explains why a metric scored a question low, via GPT-4o with
    instructor-enforced structured output (guaranteed to parse into Diagnosis, no manual
    JSON parsing/retry logic needed).
    """

    def __init__(self, model: str | None = None):
        settings = get_settings()
        if not settings.openai_api_key or len(settings.openai_api_key) < 20:
            raise RuntimeError(
                "OPENAI_API_KEY is not set (or still the .env.example placeholder) -- "
                "required for the judge. Add a real key to .env."
            )
        self._client = instructor.from_openai(OpenAI(api_key=settings.openai_api_key))
        self._model = model or settings.judge_model

    def judge(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        metric_name: str,
        score: float,
    ) -> Diagnosis:
        prompt = build_judge_prompt(
            question=question, contexts=contexts, answer=answer, metric_name=metric_name, score=score
        )
        return self._client.chat.completions.create(
            model=self._model,
            response_model=Diagnosis,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
