"""
Embedder - Generates vector embeddings from text.

Uses sentence-transformers (free, runs locally).
Model: all-MiniLM-L6-v2 (384 dimensions, fast, high quality)
"""

import logging
import os
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps sentence-transformers for text embedding generation.
    Falls back to a simple TF-IDF style if transformers not available.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._dimension = 384

    def load(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            # Disable Xet storage to avoid CDN timeout issues
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            # Use model name directly — downloads from HuggingFace if not cached
            self.model = SentenceTransformer(self.model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Embedding model loaded. Dimension: {self._dimension}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Using fallback embedder. "
                "Install with: pip install sentence-transformers"
            )
            self.model = FallbackEmbedder()
            self._dimension = 384

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if self.model is None:
            raise RuntimeError("Embedder not loaded. Call load() first.")
        if isinstance(self.model, FallbackEmbedder):
            return self.model.embed(text)
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].astype(np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts efficiently."""
        if self.model is None:
            raise RuntimeError("Embedder not loaded. Call load() first.")
        if isinstance(self.model, FallbackEmbedder):
            return np.array([self.model.embed(t) for t in texts], dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.astype(np.float32)


class FallbackEmbedder:
    """
    Simple TF-IDF + random projection fallback when sentence-transformers
    is not available. Lower quality but functional.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._vocab: dict[str, int] = {}
        self._projection: Optional[np.ndarray] = None
        self._rng = np.random.RandomState(42)

    def _get_projection(self, vocab_size: int) -> np.ndarray:
        if self._projection is None or self._projection.shape[0] != vocab_size:
            self._projection = self._rng.randn(vocab_size, self.dimension).astype(np.float32)
            norms = np.linalg.norm(self._projection, axis=0, keepdims=True)
            self._projection /= np.clip(norms, 1e-8, None)
        return self._projection

    def _tokenize(self, text: str) -> list[str]:
        import re
        tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return tokens

    def embed(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dimension, dtype=np.float32)

        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)

        vocab_size = len(self._vocab)
        bow = np.zeros(vocab_size, dtype=np.float32)
        for token in tokens:
            if token in self._vocab:
                bow[self._vocab[token]] += 1.0

        bow /= (np.sum(bow) + 1e-8)

        proj = self._get_projection(vocab_size)
        embedding = bow @ proj

        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding /= norm

        return embedding.astype(np.float32)