from langgraph.graph import StateGraph, END

from agent.nodes import AgenticRAGPipeline
from agent.state import RAGState

MAX_RETRIEVAL_ROUNDS = 2

def should_followup(state: RAGState) -> str:
    check = state["evidence_check"]
    retrieved_round = state.get("retrieval_round", 1)

    if (
        check.needs_followup_search and 
        retrieved_round < MAX_RETRIEVAL_ROUNDS and
        check.refined_search_requests
    ):
        return "followup_retrieve"
    
    else:
        return "generate_answer"


def build_rag_graph(agent: AgenticRAGPipeline) -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("plan_retrieval", agent.plan_retrieval)
    graph.add_node("retrieve_chunks", agent.retrieve_chunks)
    graph.add_node("check_evidence", agent.check_evidence)
    graph.add_node("followup_retrieve", agent.followup_retrieve)
    graph.add_node("generate_answer", agent.generate_answer)

    graph.set_entry_point("plan_retrieval")

    graph.add_edge("plan_retrieval", "retrieve_chunks")
    graph.add_edge("retrieve_chunks", "check_evidence")

    graph.add_conditional_edges(
        "check_evidence",
        should_followup,
        {
            "followup_retrieve": "followup_retrieve",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("followup_retrieve", "retrieve_chunks")
    graph.add_edge("generate_answer", END)

    return graph.compile()

