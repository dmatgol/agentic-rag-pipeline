from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        min_chunk_size: int = 200,
        table_context_max_chars: int = 600,
    ) -> None:
        """Initialize the DocumentChunker.

        Args:
        -----
        chunk_size: The size of each chunk.
        chunk_overlap: The overlap between chunks.
        min_chunk_size: Splits shorter than this are merged into the preceding split
            to prevent standalone noise chunks from being indexed separately.
        table_context_max_chars: Maximum number of characters taken from the most
            recent non-table text and prepended to every table chunk as a header.
            This gives the dense embedder the semantic context it needs to understand
            what a raw markdown table is about without bloating the chunk excessively.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.min_chunk_size = min_chunk_size
        self.table_context_max_chars = table_context_max_chars

    def _merge_tiny_splits(self, splits: list[str]) -> list[str]:
        """Merge splits shorter than min_chunk_size into their preceding split.

        This prevents standalone noise chunks such as page footers
        from being indexed on their own, where they would pollute retrieval results.  
        A tiny split that has no predecessor (i.e. it is the only split) is kept as-is 
        so we never lose content.
        """
        if not splits:
            return splits
        merged: list[str] = []
        for split in splits:
            if merged and len(split.strip()) < self.min_chunk_size:
                merged[-1] = merged[-1] + "\n\n" + split
            else:
                merged.append(split)
        return merged

    def _flush_accumulated_text(
        self,
        page_text_block: list[str],
        chunks: list[dict[str, Any]],
        document_metadata: dict[str, Any],
        page: dict[str, Any],
        chunk_idx: int,
    ) -> int:
        """Emit splitter chunks for all pending non-table text and clear the list.

        Called only before a table or at end-of-page — not after each text block,
        so consecutive text blocks keep accumulating until those boundaries.
        """
        if not page_text_block:
            return chunk_idx
        combined = "\n\n".join(page_text_block)
        page_text_block.clear()
        splits = self._merge_tiny_splits(self.text_splitter.split_text(combined))
        for split_text in splits:
            chunk_idx += 1
            chunks.append(
                self._build_chunk(
                    text=split_text,
                    document_metadata=document_metadata,
                    page=page,
                    chunk_index=chunk_idx,
                )
            )
        return chunk_idx

    def _extract_table_context(self, page_text_block: list[str]) -> str:
        """Return a trimmed header context string from the accumulated text blocks.

        We take the tail of the combined text (up to table_context_max_chars) so
        that the most immediately relevant description (e.g. "Net sales disaggregated
        by significant products and services … were as follows (in millions):") is
        always included even when the preceding text block is long.
        """
        if not page_text_block:
            return ""
        combined = "\n\n".join(page_text_block).strip()
        if len(combined) <= self.table_context_max_chars:
            return combined
        # Trim to the last N chars, starting from a clean line boundary if possible.
        tail = combined[-self.table_context_max_chars :]
        newline_pos = tail.find("\n")
        if newline_pos > 0:
            tail = tail[newline_pos:].lstrip("\n")
        return tail

    def chunk_pages(self, pages: list[dict[str, Any]], document_metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Chunk the pages of a document.

        Args:
            pages: The pages to chunk.
            document_metadata: The metadata of the document.
        """
        chunks = []
        chunk_idx = 0
        for page in pages:
            page_text_block: list[str] = []
            
            # Tracks the most recent non-table text 
            # seen on this page so we can prepend it to every 
            # table chunk as a semantic header.
            last_text_context: str = ""

            for block in page["blocks"]:
                block_type = block["type"]
                block_text = block["text"]

                # If no text, ignore the block.
                if not block_text.strip():
                    continue

                if block_type == "table":
                    if page_text_block:
                        combined = "\n\n".join(page_text_block).strip()
                        last_text_context = self._extract_table_context(page_text_block)

                        if len(combined) > self.table_context_max_chars:
                            # Preceding text is long — flush the head as regular chunks.
                            # The tail is already captured in last_text_context.
                            chunk_idx = self._flush_accumulated_text(
                                page_text_block, chunks, document_metadata, page, chunk_idx
                            )
                        else:
                            # Preceding text fits entirely as context — absorb it into the
                            # table chunk and don't emit it as a separate chunk.
                            page_text_block.clear()

                    table_text = last_text_context + "\n\n" + block_text if last_text_context else block_text

                    # Don't split table blocks (reasoning is explained in the README.md)
                    chunk_idx += 1
                    chunks.append(
                        self._build_chunk(
                            text=table_text,
                            document_metadata=document_metadata,
                            page=page,
                            chunk_index=chunk_idx,
                        )
                    )

                else:
                    page_text_block.append(block_text)

            # Trailing text after last table (or page with only non-table blocks).
            if page_text_block:
                combined = "\n\n".join(page_text_block).strip()
                if chunks and len(combined) < self.min_chunk_size:
                    # Tiny trailing text append to the last chunk 
                    # rather than emitting a standalone noise chunk.
                    chunks[-1]["text"] = chunks[-1]["text"] + "\n\n" + combined
                    page_text_block.clear()
                else:
                    chunk_idx = self._flush_accumulated_text(
                        page_text_block, chunks, document_metadata, page, chunk_idx
                    )

        return chunks

    def _build_chunk(
        self,
        text: str,
        document_metadata: dict[str, Any],
        page: dict[str, Any],
        chunk_index: int,
    ) -> dict[str, Any]:
        return {
            "chunk_id": self._build_chunk_id(
                document_metadata=document_metadata,
                page_number=page["page_number"],
                chunk_index=chunk_index,
            ),
            "text": text.strip(),
            "metadata": {
                **document_metadata,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
            },
        }

    def _build_chunk_id(
        self,
        document_metadata: dict[str, Any],
        page_number: int,
        chunk_index: int,
    ) -> str:
        ticker = document_metadata.get("company_ticker", "UNKNOWN_TICKER")
        year = document_metadata.get("year", "UNKNOWN_YEAR")
        quarter = document_metadata.get("quarter", "UNKNOWN_QUARTER")


        return f"{ticker}_{year}_{quarter}_p{page_number}_cidx{chunk_index}"