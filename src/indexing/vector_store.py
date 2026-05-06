import uuid
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import models


class QdrantVectorStore:
    """Qdrant Vector Store."""
    def __init__(
        self,
        collection_name: str,
        dense_vector_size: int,
        qdrant_url: str = "http://localhost:6333",
        sparse_model_name: str = "Qdrant/bm25",
        recreate_collection: bool = False,
    ) -> None:
        """Initialize the QdrantVectorStore.

        Args:
        -----
        collection_name: The name of the collection.
        dense_vector_size: The size of the dense vector.
        qdrant_url: The URL of the Qdrant server.
        sparse_model_name: The name of the sparse model.
        recreate_collection: Whether to recreate the collection.
        """
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        self._ensure_collection(
            dense_vector_size=dense_vector_size,
            recreate_collection=recreate_collection,
        )

    def _ensure_collection(
        self,
        dense_vector_size: int,
        recreate_collection: bool,
    ) -> None:
        """Ensure the collection exists."""
        if self.client.collection_exists(self.collection_name):
            if recreate_collection:
                self.client.delete_collection(self.collection_name)
            else:
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )
        self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        """Create payload indexes for fields commonly used in filters."""
        indexes = {
            "ticker": models.PayloadSchemaType.KEYWORD,
            "year": models.PayloadSchemaType.KEYWORD,
            "quarter": models.PayloadSchemaType.KEYWORD,
        }

        for field_name, field_schema in indexes.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception:
                # Index may already exist. Safe to ignore for this assignment.
                pass

    def upsert_chunks(self, embedded_chunks: list[dict[str, Any]], batch_size: int = 64) -> None:
        """Upsert the chunks into the vector store."""
        for start in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[start:start+batch_size]

            texts = [chunk["text"] for chunk in batch]
            sparse_vectors = list(self.sparse_model.embed(texts))

            points = []
            for chunk, sparse_vector in zip(batch, sparse_vectors):
                points.append(
                    models.PointStruct(
                        id=self._to_qdrant_id(chunk["chunk_id"]),
                        vector={
                            "dense": chunk["embedding"],
                            "sparse": models.SparseVector(
                                indices=sparse_vector.indices.tolist(),
                                values=sparse_vector.values.tolist(),
                            ),
                        },
                        payload=self._build_payload(chunk),
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def _to_qdrant_id(self, chunk_id: str) -> str:
        """Convert the chunk ID to a Qdrant ID."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _build_payload(self, chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            **chunk.get("metadata", {}),
        }

    def dense_search(self, query_embedding, limit=3, ticker=None):
        query_filter = None

        if ticker:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="company_ticker",
                        match=models.MatchValue(value=ticker),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using="dense",
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return results.points