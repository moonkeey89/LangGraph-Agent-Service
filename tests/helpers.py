from langchain_core.embeddings import Embeddings


TEST_EMBEDDING_DIMENSIONS = 16


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
