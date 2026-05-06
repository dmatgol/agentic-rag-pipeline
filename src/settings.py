from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Load secrets and config from environment / ``.env`` at the project root."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # APIs 
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    llama_cloud_api_key: str | None = Field(default=None, description="LlamaCloud / LlamaParse API key")

    # Model Specification
    vlm_model: str = "gpt-5-mini"
    llama_parse_model: str = "llama-parse"
    openai_embedding_model: str = "text-embedding-3-small"
    agentic_rag_model: str = "gpt-5-mini"
    evaluation_judge_model: str = "gpt-5-mini"

    # Qdrant
    qdrant_url: str = "http://localhost:6333" 
    collection_name: str = "financial_reports"
    recreate_collection_on_full_ingest: bool = True


settings = Settings()
