from typing import Any

from openai import OpenAI


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    def embed_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 64,
    ) -> list[dict[str, Any]]:
        embedded_chunks = []

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk["text"] for chunk in batch]

            embeddings = self.embed_texts(texts)

            for chunk, embedding in zip(batch, embeddings):
                embedded_chunks.append(
                    {
                        **chunk,
                        "embedding": embedding,
                    }
                )

        return embedded_chunks