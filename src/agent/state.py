from typing import TypedDict

from agent.models import RetrievalPlan, RetrievedChunk, EvidenceCheck, FinalAnswer


class RAGState(TypedDict, total=False):
    question: str

    retrieval_plan: RetrievalPlan
    retrieved_chunks: list[RetrievedChunk]
    evidence_check: EvidenceCheck

    retrieval_round: int
    final_answer: FinalAnswer