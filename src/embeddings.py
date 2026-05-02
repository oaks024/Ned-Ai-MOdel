"""Local embedding model wrapper.

We use sentence-transformers so embeddings are free and work offline after
the first download. all-MiniLM-L6-v2 produces 384-dim vectors and is a
solid default for English-language admission text.
"""
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        # normalize_embeddings=True lets us use cosine similarity directly.
        vectors = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()
