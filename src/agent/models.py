from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="The query to search for.")
    tickers: list[str] = Field(default_factory=list, description="The companies to search for.")
    years: list[int] = Field(default_factory=list, description="The years to search for.")
    quarters: list[str] = Field(default_factory=list, description="The quarters to search for.")
    limit: int = Field(default=30, description="The number of results to return.")


class DocumentSearchInput(BaseModel):
    query: str = Field(
        description="Search query to retrieve relevant financial document chunks."
    )

    tickers: list[str] = Field(
        default_factory=list,
        description="Optional company tickers, e.g. ['AAPL', 'MSFT']."
    )

    years: list[int] = Field(
        default_factory=list,
        description="Optional document years to filter by, e.g. [2023]."
    )

    quarters: list[str] = Field(
        default_factory=list,
        description="Optional document quarters to filter by, e.g. ['Q1', 'Q2']."
    )

    limit: int = Field(
        default=8,
        description="Maximum number of chunks to retrieve when no company fanout is needed."
    )

    limit_per_company: int = Field(
        default=5,
        description="Maximum number of chunks to retrieve per company when multiple companies are provided."
    )


class RetrievalPlan(BaseModel):
    retrieval_strategy: Literal[
        "single_company",
        "multi_company_fanout",
        "all_companies_fanout",
        "global_search",
        "ambiguous_global_search",
    ]
    target_companies: list[str] = Field(default_factory=list, description="The companies to search for.")
    target_quarters: list[str] = Field(default_factory=list, description="The quarters to search for.")
    target_years: list[str] = Field(default_factory=list, description="The years to search for.")
    
    search_requests: list[SearchRequest] = Field(default_factory=list, description="The search requests to perform.")

    reasoning_summary: str


class EvidenceCheck(BaseModel):
    is_context_sufficient: bool
    needs_followup_search: bool
    missing_information: list[str] = Field(default_factory=list)
    refined_search_requests: list[SearchRequest] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: dict[str, Any]


class FinalAnswer(BaseModel):
    answer: str
    companies_covered: list[str]
    supporting_evidence: list[RetrievedChunk]
    sources: list[dict[str, Any]]
    confidence_score: Literal["low", "medium", "high"]


def serialize_graph_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert LangGraph result into JSON-friendly dict."""

    def serialize(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, list):
            return [serialize(v) for v in value]
        if isinstance(value, dict):
            return {k: serialize(v) for k, v in value.items()}
        return value

    return {
        "retrieval_plan": serialize(result.get("retrieval_plan")),
        "retrieved_chunks": serialize(result.get("retrieved_chunks", [])),
        "evidence_check": serialize(result.get("evidence_check")),
        "final_answer": serialize(result.get("final_answer")),
    }