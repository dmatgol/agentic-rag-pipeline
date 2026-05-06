from fastembed import SparseTextEmbedding
from qdrant_client.http import models
from indexing.vector_store import QdrantVectorStore
from indexing.embedding import OpenAIEmbedder
from typing import Any

from agent.models import RetrievedChunk


def sort_retrieved_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(
        chunks,
        key=lambda c: (
            c.metadata.get("company_ticker") or "",
            c.metadata.get("source_file") or c.metadata.get("file_name") or c.metadata.get("source_path") or "",
            c.metadata.get("chunk_index", c.metadata.get("chunk_idx", 10**9)),
        ),
    )


class DocumentSearchTool:
    def __init__(self, vector_store: QdrantVectorStore, embedder: OpenAIEmbedder) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def search(
        self, 
        query: str,
        tickers: list[str] | None = None,
        years: list[int] | None = None,
        quarters: list[str] | None = None,
        limit: int = 15,
        limit_per_company: int = 8,
    ) -> list[RetrievedChunk]:
        """Search the vector store for relevant chunks."""
        tickers = tickers or []
        years = years or []
        quarters = quarters or []

        query_dense_embedding = self.embedder.embed_texts([query])[0]
        sparse_query = list(self.sparse_model.embed([query]))[0]

        if len(tickers) > 1:
            all_chunks: list[RetrievedChunk] = []

            for ticker in tickers:
                chunks = self._search_for_chunks(
                    query_dense_embedding=query_dense_embedding,
                    sparse_query=sparse_query,
                    tickers=[ticker],
                    years=years,
                    quarters=quarters,
                    limit=limit_per_company,
                )
                all_chunks.extend(chunks)

            return self._dedupe_and_sort(all_chunks)
        else:
            chunks = self._search_for_chunks(
                query_dense_embedding=query_dense_embedding,
                sparse_query=sparse_query,
                tickers=tickers,
                years=years,
                quarters=quarters,
                limit=limit,
            )
            return self._dedupe_and_sort(chunks)

    def _search_for_chunks(
        self,
        query_dense_embedding: list[float],
        sparse_query: list[float],
        tickers: list[str],
        years: list[int],
        quarters: list[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        filters = self._build_filters(
            tickers=tickers,
            years=years,
            quarters=quarters,
        )

        # Fetch a larger candidate pool for each leg so that RRF re-ranking
        # operates on more options before cutting to the requested `limit`.
        prefetch_limit = max(limit * 3, 40)
        prefetch = [
            models.Prefetch(
                query=query_dense_embedding,
                using="dense",
                limit=prefetch_limit,
                filter=filters,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_query.indices.tolist(),
                    values=sparse_query.values.tolist(),
                ),
                using="sparse",
                limit=prefetch_limit,
                filter=filters,
            ),
        ]

        results = self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )
        return [
            self._to_retrieved_chunk(
                {
                    "score": point.score,
                    "payload": point.payload,
                }
            )
            for point in results.points
        ]

    
    def _build_filters(
        self,
        tickers: list[str],
        years: list[int],
        quarters: list[str],
    ) -> dict[str, Any]:
        """Build the filters for the search."""
        must_conditions: list[models.FieldCondition] = []

        if tickers:
            tickers = [ticker.upper() for ticker in tickers]
            must_conditions.append(
                models.FieldCondition(key="company_ticker", match=models.MatchAny(any=tickers))
            )
        
        if years:
            must_conditions.append(
                models.FieldCondition(key="year", match=models.MatchAny(any=[str(year) for year in years]))
            )

        if quarters:
            must_conditions.append(
                models.FieldCondition(key="quarter", match=models.MatchAny(any=[quarter.upper() for quarter in quarters]))
            )
        
        if not must_conditions:
            return None
        
        return models.Filter(must=must_conditions)

    def _to_retrieved_chunk(
        self,
        result: dict[str, Any],
    ) -> RetrievedChunk:
        payload = result["payload"]

        return RetrievedChunk(
            text=payload["text"],
            score=float(result["score"]),
            metadata={
                key: value
                for key, value in payload.items()
                if key != "text"
            },
        )

    def _dedupe_and_sort(
        self,
        chunks: list[RetrievedChunk],   
    ) -> list[RetrievedChunk]:  
        seen = set()
        deduped: list[RetrievedChunk] = []

        for chunk in chunks:
            chunk_id = chunk.metadata.get("chunk_id")

            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            deduped.append(chunk)

        return sort_retrieved_chunks(deduped)
