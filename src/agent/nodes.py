import json
from openai import OpenAI

from settings import settings
from agent.tools import DocumentSearchTool, sort_retrieved_chunks
from agent.prompts import ANSWER_GENERATION_SYSTEM_PROMPT, EVIDENCE_CHECK_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT
from agent.state import RAGState
from agent.models import EvidenceCheck, FinalAnswer, RetrievalPlan, RetrievedChunk



class AgenticRAGPipeline:
    def __init__(
        self,
        llm_model: str,
        openai_api_key: str,
        search_tool: DocumentSearchTool,
        max_retrieval_rounds: int = 3,
        max_chunks_per_company: int = 15,
    ) -> None:
        self.client = OpenAI(api_key=openai_api_key)
        self.llm_model = llm_model
        self.search_tool = search_tool
        self.max_retrieval_rounds = max_retrieval_rounds
        self.max_chunks_per_company = max_chunks_per_company

    def _cap_per_company(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Keep only the top max_chunks_per_company chunks for each company ticker."""
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            ticker = chunk.metadata.get("company_ticker") or "_unknown"
            grouped.setdefault(ticker, []).append(chunk)

        capped = []
        for ticker_chunks in grouped.values():
            top = sorted(ticker_chunks, key=lambda c: c.score, reverse=True)
            capped.extend(top[: self.max_chunks_per_company])
        return capped

    def plan_retrieval(self, state: RAGState) -> RAGState:
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = {
            "question": state["question"],
            "required_schema": RetrievalPlan.model_json_schema(),
        }

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            response_format={"type": "json_object"},
        )

        plan = RetrievalPlan.model_validate_json(
            response.choices[0].message.content
        )

        state["retrieval_plan"] = plan
        state["retrieval_round"] = 1
        return state

    def retrieve_chunks(self, state: RAGState) -> RAGState:
        plan = state["retrieval_plan"]

        all_chunks: list[RetrievedChunk] = []

        for request in plan.search_requests:
            chunks = self.search_tool.search(
                query=request.query,
                tickers=request.tickers,
                years=request.years,
                quarters=request.quarters,
                limit=request.limit,
            )
            all_chunks.extend(chunks)

        # Deduplicate chunks
        seen = set()
        deduped = []

        for chunk in all_chunks:
            chunk_id = chunk.metadata.get("chunk_id")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            deduped.append(chunk)

        state["retrieved_chunks"] = sort_retrieved_chunks(self._cap_per_company(deduped))
        return state


    def check_evidence(self, state: RAGState) -> RAGState:

        system_prompt = EVIDENCE_CHECK_SYSTEM_PROMPT
        chunks_for_prompt = [
            {
                "chunk_id": c.metadata.get("chunk_id"),
                "ticker": c.metadata.get("ticker"),
                "source_file": c.metadata.get("source_file"),
                "page_number": c.metadata.get("page_number"),
                "chunk_index": c.metadata.get("chunk_index", c.metadata.get("chunk_idx")),
                "text": c.text,
            }
            for c in state.get("retrieved_chunks", [])
        ]

        user_prompt = {
            "question": state["question"],
            #"retrieval_plan": state["retrieval_plan"].model_dump(),
            "retrieved_chunks": chunks_for_prompt,
            "retrieval_round": state.get("retrieval_round", 1),
            "max_retrieval_rounds": self.max_retrieval_rounds,
            "required_schema": EvidenceCheck.model_json_schema(),
        }

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            response_format={"type": "json_object"},
        )

        evidence_check = EvidenceCheck.model_validate_json(
            response.choices[0].message.content
        )

        state["evidence_check"] = evidence_check
        return state

    def followup_retrieve(self, state: RAGState) -> RAGState:
        evidence_check = state["evidence_check"]

        if not evidence_check.needs_followup_search:
            return state

        followup_chunks: list[RetrievedChunk] = []

        for request in evidence_check.refined_search_requests:
            chunks = self.search_tool.search(
                query=request.query,
                tickers=request.tickers,
                years=request.years,
                quarters=request.quarters,
                limit=request.limit,
            )
            followup_chunks.extend(chunks)

        all_chunks = state.get("retrieved_chunks", []) + followup_chunks

        seen = set()
        deduped = []

        for chunk in all_chunks:
            chunk_id = chunk.metadata.get("chunk_id")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            deduped.append(chunk)

        state["retrieved_chunks"] = sort_retrieved_chunks(self._cap_per_company(deduped))
        state["retrieval_round"] = state.get("retrieval_round", 1) + 1
        return state

    def generate_answer(self, state: RAGState) -> RAGState:
        system_prompt = ANSWER_GENERATION_SYSTEM_PROMPT
        chunks_for_prompt = [
            {
                "chunk_id": c.metadata.get("chunk_id"),
                "ticker": c.metadata.get("ticker"),
                "source_file": c.metadata.get("source_file"),
                "page_number": c.metadata.get("page_number"),
                "chunk_index": c.metadata.get("chunk_index", c.metadata.get("chunk_idx")),
                "text": c.text,
            }
            for c in state.get("retrieved_chunks", [])
        ]

        user_prompt = {
            "question": state["question"],
            "retrieved_chunks": chunks_for_prompt,
            "required_schema": FinalAnswer.model_json_schema(),
        }

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            response_format={"type": "json_object"},
        )

        final_answer = FinalAnswer.model_validate_json(
            response.choices[0].message.content
        )

        state["final_answer"] = final_answer
        return state