from __future__ import annotations

import json
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field



class EvaluationJudgement(BaseModel):
    score: int = Field(ge=0, le=100)
    verdict: Literal["correct", "incorrect"]

    factual_correctness: int = Field(ge=0, le=25)
    groundedness: int = Field(ge=0, le=25)
    completeness: int = Field(ge=0, le=25)
    retrieved_quality: int = Field(ge=0, le=25)

    rationale: str = Field(description="The rationale for the verdict")


class RAGEvaluator:

    def __init__(self, openai_api_key: str, judge_model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI(api_key=openai_api_key)
        self.judge_model = judge_model

    def evaluate(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        retrieved_chunks: list[dict[str, Any]],
        retrieved_plan: dict[str, Any] | None = None,
        evidence_check: dict[str, Any] | None = None,
    ) -> EvaluationJudgement:
        system_prompt = evaluation_system_prompt()
        
        user_payload = {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_plan": retrieved_plan,
            "evidence_check": evidence_check,
            "required_schema": EvaluationJudgement.model_json_schema(),
        }

        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            response_format={"type": "json_object"},
        )

        return EvaluationJudgement.model_validate_json(
            response.choices[0].message.content
        )

def evaluation_system_prompt() -> str:
    return """You are an evaluator for a financial-document RAG system.

Compare the generated answer against the expected answer and retrieved evidence.

Score from 0 to 100.

Evaluation criteria:
1. Factual correctness: Does the generated answer match the expected answer?
2. Groundedness: Is the answer supported by the retrieved context?
3. Completeness: Does it answer all parts of the question?
4. Retrieved quality: Did the retrieved context contain relevant sources?

Each of the above criteria is scored from 0 to 25. The total score is the sum of the scores for each criterion.

Rules:
- Be strict with numeric values.
- Penalize unsupported claims.
- Penalize answers that use the wrong company, year, quarter, or metric.
- If the expected answer is not fully supported by retrieved context, mention that.
- Return valid JSON only.
"""
