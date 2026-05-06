import streamlit as st

from indexing.embedding import OpenAIEmbedder
from indexing.vector_store import QdrantVectorStore
from agent.tools import DocumentSearchTool
from agent.graph import build_rag_graph
from agent.nodes import AgenticRAGPipeline
from settings import REPO_ROOT, settings
from agent.models import serialize_graph_result
from evaluation.evaluator import RAGEvaluator
import json


@st.cache_resource
def load_rag_app():
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in .env")

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )

    # text-embedding-3-small default dimension
    dense_vector_size = 1536

    vector_store = QdrantVectorStore(
        collection_name=settings.collection_name,
        dense_vector_size=dense_vector_size,
        qdrant_url=settings.qdrant_url,
        recreate_collection=False,
    )

    search_tool = DocumentSearchTool(
        vector_store=vector_store,
        embedder=embedder,
    )

    agent = AgenticRAGPipeline(
        llm_model=settings.agentic_rag_model,
        openai_api_key=settings.openai_api_key,
        search_tool=search_tool,
    )

    graph = build_rag_graph(agent)

    return graph


@st.cache_resource
def load_evaluator():
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in .env")

    return RAGEvaluator(
        openai_api_key=settings.openai_api_key,
        judge_model=settings.evaluation_judge_model,
    )

@st.cache_data
def load_evaluation_questions() -> list[dict]:
    with open(REPO_ROOT / "src/evaluation/evaluation_questions.json", "r") as f:
        return json.load(f)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("Sources"):
        for i, source in enumerate(sources, start=1):
            st.markdown(f"**Source {i}**")
            st.write(
                {
                    "ticker": source.get("ticker"),
                    "company": source.get("company"),
                    "source_file": source.get("source_file"),
                    "page_number": source.get("page_number"),
                    "chunk_idx": source.get("chunk_idx"),
                    "chunk_id": source.get("chunk_id"),
                }
            )


def render_retrieved_context(retrieved_chunks: list[dict]) -> None:
    """Show each retrieved chunk in full (text, score, metadata) — no truncation."""
    with st.expander("Retrieved context", expanded=True):
        for i, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(f"**Chunk {i}**")
            st.code(
                json.dumps(chunk, ensure_ascii=False, indent=2),
                language="json",
            )


def run_graph_for_question(graph, question: str) -> dict:

    result = graph.invoke(
        {
            "question": question,
        }
    )

    return serialize_graph_result(result)


def rag_mode(graph):
    st.subheader("Rag")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    user_question = st.chat_input("Ask a question about the indexed financial documents...")

    if not user_question:
        return

    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Running agentic RAG..."):
            result = run_graph_for_question(graph, user_question)
            final_answer = result["final_answer"]

            answer_text = final_answer.get("answer", "")
            sources = final_answer.get("sources", [])

            st.markdown(answer_text)
            render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
        }
    )


def evaluation_mode(graph, evaluator):
    st.subheader("Evaluation")

    questions = load_evaluation_questions()

    selected = st.selectbox(
        "Select an evaluation question",
        questions,
        format_func=lambda q: f"{q['id']} — {q['question']}",
    )

    st.markdown("### Expected answer")
    st.info(selected["expected_answer"])

    with st.expander("Evaluation metadata", expanded=False):
        st.json(
            {
                "expected_companies": selected.get("expected_companies"),
                "expected_years": selected.get("expected_years"),
                "expected_quarters": selected.get("expected_quarters"),
                "expected_metric": selected.get("expected_metric"),
                "question_type": selected.get("question_type"),
            }
        )

    if not st.button("Run evaluation", type="primary"):
        return

    with st.spinner("Running agentic RAG pipeline..."):
        result = run_graph_for_question(graph, selected["question"])

    final_answer = result["final_answer"]
    generated_answer = final_answer.get("answer", "")

    st.markdown("## Agent trace")

    st.markdown("### 1. User question")
    st.write(selected["question"])

    st.markdown("### 2. Retrieval plan")
    st.json(result.get("retrieval_plan"))

    st.markdown("### 3. Retrieved context")
    render_retrieved_context(result.get("retrieved_chunks", []))

    st.markdown("### 4. Evidence check")
    st.json(result.get("evidence_check"))

    st.markdown("### 5. Generated answer")
    st.success(generated_answer)

    st.markdown("### 6. Sources of the answer")
    st.json(result.get("final_answer"))


    st.markdown("## LLM-as-judge evaluation")

    with st.spinner("Judging generated answer..."):
        judgement = evaluator.evaluate(
            question=selected["question"],
            expected_answer=selected["expected_answer"],
            generated_answer=generated_answer,
            retrieved_chunks=result.get("retrieved_chunks", []),
            retrieved_plan=result.get("retrieval_plan"),
            evidence_check=result.get("evidence_check"),
        )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Score", f"{judgement.score}/100")
    col2.metric("Factual", f"{judgement.factual_correctness}/25")
    col3.metric("Grounded", f"{judgement.groundedness}/25")
    col4.metric("Complete", f"{judgement.completeness}/25")
    col5.metric("Retrieved", f"{judgement.retrieved_quality}/25")

    if judgement.verdict == "correct":
        st.success(f"Verdict: {judgement.verdict}")
    else:
        st.error(f"Verdict: {judgement.verdict}")

    st.markdown("### Judge rationale")
    st.write(judgement.rationale)

    with st.expander("Raw judgement JSON"):
        st.json(judgement.model_dump())


def main():
    st.set_page_config(
        page_title="Financial RAG Assistant",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Financial Document RAG Assistant")

    st.caption(
        "Ask questions over indexed investor-relations documents. "
        "The assistant uses metadata-aware hybrid retrieval and structured answers."
    )

    graph = load_rag_app()
    evaluator = load_evaluator()

    mode = st.sidebar.radio("Mode", ["RAG", "Evaluation"])

    if mode == "RAG":
        rag_mode(graph)
    elif mode == "Evaluation":
        evaluation_mode(graph, evaluator)


if __name__ == "__main__":
    main()