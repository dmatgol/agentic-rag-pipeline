from pathlib import Path
from typing import Any

import fitz
import tempfile
from llama_cloud import LlamaCloud
import json
from openai import OpenAI
from settings import settings


class DocumentParser:
    def __init__(self, use_vlm: bool = False, llama_cloud_api_key: str | None = None) -> None:
        """Initialize the DocumentParser.

        Args:
            use_vlm: Whether to use VLM for document parsing.
            llama_cloud_api_key: Override; defaults to ``settings.llama_cloud_api_key`` (env ``LLAMA_CLOUD_API_KEY``).
        """
        self.use_vlm = use_vlm
        api_key = llama_cloud_api_key or settings.llama_cloud_api_key
        if not api_key:
            raise ValueError(
                "LlamaCloud API key missing: set LLAMA_CLOUD_API_KEY in .env or pass llama_cloud_api_key=..."
            )
        self.llama_parse_client = LlamaCloud(api_key=api_key)
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

    def parse_document(self, document_path: Path) -> list[dict[str, Any]]:
        """First Parse of the document using LlamaParse.

        Args:
            document_path: The path to the document to parse.

        Returns:
            Page-level records with metadata.
        """
        file = self.llama_parse_client.files.create(file=document_path, purpose="parse")
        response = self.llama_parse_client.parsing.parse(
            file_id=file.id,
            tier="cost_effective",
            version="latest",
            expand=["markdown", "items", "text"]
        )

        pages = []
        for page_number, page in enumerate(response.items.pages):
            contains_table = self._contains_item_type(page, {"table"})
            contains_image = self._contains_item_type(page, {"image", "figure", "image"})
            if (contains_table or contains_image) and self.use_vlm:
                page_markdown = response.markdown.pages[page_number].markdown
                page_image = self._render_page_to_image(document_path, page_number)
                parsed_text = self._parse_page_with_vlm(page_markdown, page_image)
                blocks = [
                    {
                        "type": "table" if contains_table else "text",
                        "text": parsed_text,
                    }
                ]
            else:
                parsed_text = response.markdown.pages[page_number].markdown
                blocks = self._extract_blocks_from_llamaparse_page(page)
            pages.append({
                "page_number": page_number,
                "text": parsed_text,
                "blocks": blocks,
                "contains_table": contains_table,
                "contains_image": contains_image,
                "needs_vlm": contains_table or contains_image,
            })
        return pages

    
    def _extract_blocks_from_llamaparse_page(self, page: Any) -> list[dict[str, Any]]:
        """Extract blocks from a LlamaParse page.

        Args:
            page: The page to extract blocks from.

        Returns:
            List of blocks.
        """
        blocks = []

        for item in getattr(page, "items", []) or []:
            item_type = getattr(item, "type", None)

            if item_type == "table":
                text = self._get_table_text(item)
                block_type = "table"
            else:
                text = self._get_text_item_text(item)
                block_type = "text"

            if text and text.strip():
                blocks.append(
                    {
                        "type": block_type,
                        "text": text.strip(),
                    }
                )

        return blocks

    def _contains_item_type(self, page: Any, item_types: set[str]) -> bool:
        return any(
            getattr(item, "type", None) in item_types
            for item in getattr(page, "items", []) or []
        )

    def _get_table_text(self, item: Any) -> str:
        return (
            getattr(item, "md", None)
            or getattr(item, "csv", None)
            or getattr(item, "html", None)
            or ""
        )

    def _get_text_item_text(self, item: Any) -> str:
        return (
            getattr(item, "md", None)
            or getattr(item, "value", None)
            or getattr(item, "text", None)
            or ""
        )
    
    def _parse_page_with_vlm(self, text_markdown: str, image_path: Path) -> str:
        """Render a PDF page to image and send it to the local VLM.

        Args:
            pdf_path: Path to the original PDF.
            page_number: 1-indexed page number.

        Returns:
            Clean markdown returned by the VLM.
        """
        system_prompt = """
        You are a financial document parsing assistant.

        You will receive two inputs:
        1. Text markdown of the page.
        2. Image of the page.

        # GOAL
        Your goal is to clean and format correctly the text/table content using the provided text and image.

        # PROCESS
        Process in this exact order:

        **STEP 1. TABLE FORMATTING**
        - Format tables as clean markdown and visually easily readable for a human reader.
        - Keep ALL numbers, dates and data exactly as they appear in the image.
        - When you see from the image that a row / column is a merged row/column,
        please propagate the value across all merged cells or mark as [MERGED ACROSS ROWS] or
        [MERGED ACROSS COLUMNS] for the other rows/columns.

        # CRITICAL RULES
        - Never skip or summarize content.
        - Never return empty cleaned text.
        - Use the provided image as the primary source of truth.

        # OUTPUT
        Return only the cleaned markdown text. NO EXTRA INFORMATION.
        """


        user_prompt = f"""
            ## GOAL
            Clean and format the text/table content using the provided per page text and image.
            
            PAGE TEXT
            {text_markdown}

            IMAGE OF THE PAGE
            {image_path}
        """
        
        response = self.openai_client.chat.completions.create(
            model=settings.vlm_model,
            messages=[
                {
                    "role": "system",  "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [str(image_path)],
                }
            ],
        )

        return response["message"]["content"].strip()

    def _render_page_to_image(self, pdf_path: Path, page_number: int) -> Path:
        """Render a single PDF page to a PNG image.

        Args:
            pdf_path: Path to the PDF file.
            page_number: 1-indexed page number.

        Returns:
            Path to the rendered image.
        """
        pdf_path = Path(pdf_path)

        with fitz.open(pdf_path) as doc:
            page = doc[page_number - 1]  # PyMuPDF is 0-indexed
            pix = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)

            tmp_dir = Path(tempfile.gettempdir())
            image_path = tmp_dir / f"{pdf_path.stem}_page_{page_number}.png"
            pix.save(str(image_path))

        return image_path

    

            