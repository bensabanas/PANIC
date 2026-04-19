"""
PANIC Encoder

Transforms raw text into dense vector embeddings for retrieval.

MVP: Off-the-shelf sentence-transformers.
Default: all-MiniLM-L12-v2 (384-dim, 12 layers — better quality than L6).
Each turn is encoded independently. Retrieval uses cosine similarity on these embeddings.

Future: Contextual encoder that conditions on recent turns before encoding.
"""

import numpy as np
from typing import Optional

# Default encoder: all-MiniLM-L6-v2 (384-dim, 6 layers).
# Chosen for fast inference and good retrieval quality.
# All eval benchmarks use this model — changing it requires re-evaluation.
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class PanicEncoder:
    """Encodes text into dense vectors for PANIC retrieval."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model = None  # lazy load

    @property
    def embedding_dim(self) -> int:
        """Dimension of output embeddings."""
        # all-MiniLM-L6-v2 outputs 384-dim vectors
        # If model changes, this must be updated
        return self._get_model().get_sentence_embedding_dimension()

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a dense vector.

        Args:
            text: Raw text input (user turn or LLM response).

        Returns:
            1-D numpy array of shape (embedding_dim,).
        """
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode multiple texts in a single batch.

        Args:
            texts: List of text strings.

        Returns:
            2-D numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        model = self._get_model()
        embeddings = model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32
        )
        return embeddings.astype(np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two embeddings.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Cosine similarity score (-1 to 1).
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def similarity_matrix(self, embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
        """
        Cosine similarity of a query against multiple embeddings.

        Args:
            embeddings: 2-D array of shape (n, embedding_dim).
            query: 1-D array of shape (embedding_dim,).

        Returns:
            1-D array of shape (n,) with similarity scores.
        """
        if len(embeddings) == 0:
            return np.empty(0, dtype=np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            return np.zeros(len(embeddings), dtype=np.float32)

        # Avoid division by zero for zero-norm embeddings
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms
        normalized_query = query / query_norm

        return (normalized @ normalized_query).astype(np.float32)
