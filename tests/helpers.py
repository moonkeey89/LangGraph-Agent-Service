from langchain_core.embeddings import Embeddings
from typing import Annotated

from fastapi import FastAPI, Header

from ai_agent_learning.api.dependencies import get_user_id


TEST_EMBEDDING_DIMENSIONS = 16


def install_test_identity(app: FastAPI) -> FastAPI:
    """Use the legacy header only inside tests through FastAPI's override API."""

    def get_test_user_id(
        x_user_id: Annotated[str, Header(alias="X-User-ID")],
    ) -> str:
        return x_user_id

    app.dependency_overrides[get_user_id] = get_test_user_id
    return app


class DeterministicTestEmbeddings(Embeddings):
    """Small dependency-free embeddings for deterministic automated tests."""

    def __init__(self, dimensions: int = TEST_EMBEDDING_DIMENSIONS):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        normalized = text.casefold().replace(" ", "")
        for index, character in enumerate(normalized):
            bucket = (ord(character) + index * 31) % self.dimensions
            vector[bucket] += 1.0

        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude:
            return [value / magnitude for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
