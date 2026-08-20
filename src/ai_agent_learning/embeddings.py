from langchain_core.embeddings import Embeddings


DEFAULT_EMBEDDING_MODEL = "minishlab/potion-multilingual-128M"
DEFAULT_EMBEDDING_DIMENSIONS = 256


class LocalModel2VecEmbeddings(Embeddings):
    """Lazy multilingual static embeddings downloaded on first real use."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from model2vec import StaticModel

            self._model = StaticModel.from_pretrained(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self._load_model().encode(texts)
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
