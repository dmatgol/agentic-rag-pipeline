from __future__ import annotations

import json
import sys
from pathlib import Path

from ingestion.chunker import DocumentChunker
from ingestion.document_parsing import DocumentParser
from indexing.embedding import OpenAIEmbedder
from indexing.vector_store import QdrantVectorStore
from settings import REPO_ROOT, settings

MANIFEST_PATH = REPO_ROOT / "src" / "ingestion" / "documents_manifest.json"


def load_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_document_metadata(record: dict) -> dict[str, str]:
    """Must align with ``DocumentChunker`` / chunk ids (company_ticker, year, quarter, …)."""
    return {
        "company_ticker": record["company_ticker"],
        "company_name": record["company_name"],
        "year": record["year"],
        "quarter": record["quarter"],
        "file_name": record["file_name"],
        "source_path": record["source_path"],
    }


def resolve_pdf_path(record: dict) -> Path:
    return (REPO_ROOT / record["source_path"]).resolve()


def main() -> None:
    if not settings.openai_api_key:
        print("Set OPENAI_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    manifest = load_manifest(MANIFEST_PATH)
    parser = DocumentParser(
        llama_cloud_api_key=settings.llama_cloud_api_key,
    )
    chunker = DocumentChunker()
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )

    all_chunks: list[dict] = []

    for record in manifest:
        pdf_path = resolve_pdf_path(record)
        if not pdf_path.is_file():
            print(f"Skip (missing file): {pdf_path}")
            continue

        document_metadata = build_document_metadata(record)
        print(f"Parsing: {pdf_path}")

        pages = parser.parse_document(pdf_path)
        chunks = chunker.chunk_pages(pages, document_metadata)
        print(f"  → {len(chunks)} chunks for {record['file_name']}")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks produced; check manifest paths and PDF files under data/.")
        sys.exit(1)

    print(f"Total chunks: {len(all_chunks)}")
    print("Embedding…")

    embedded_chunks = embedder.embed_chunks(all_chunks, batch_size=64)
    dense_vector_size = len(embedded_chunks[0]["embedding"])

    vector_store = QdrantVectorStore(
        collection_name=settings.collection_name,
        dense_vector_size=dense_vector_size,
        qdrant_url=settings.qdrant_url,
        recreate_collection=settings.recreate_collection_on_full_ingest,
    )

    print("Upserting into Qdrant…")
    vector_store.upsert_chunks(embedded_chunks, batch_size=64)

    print(
        f"Done. Indexed {len(embedded_chunks)} chunks into «{settings.collection_name}» "
        f"at {settings.qdrant_url}"
    )


if __name__ == "__main__":
    main()
