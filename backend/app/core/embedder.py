"""
Embedder - Generates vector embeddings from text.

Uses sentence-transformers locally if enabled.
Uses lightweight fallback embeddings on low-memory environments.
"""

import logging
import os
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Set to False - using lightweight model that fits in 512MB RAM
USE_LIGHTWEIGHT_EMBEDDER = False


class Embedder:
    """
    Embedding wrapper with automatic fallback support.
    """

    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2"):
        self.model_name = model_name
        self.model = None
        self._dimension = 384

    def load(self):
        """Load embedding model."""

        try:
            if USE_LIGHTWEIGHT_EMBEDDER:
                logger.warning("Using lightweight fallback embedder")
                self.model = FallbackEmbedder()
                self._dimension = 384
                return

            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            self.model = SentenceTransformer(self.model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding loaded. Dimension: {self._dimension}")

        except Exception as e:
            logger.warning(f"Error loading transformer: {e}")
            logger.warning("Switching to fallback embedder...")
            self.model = FallbackEmbedder()
            self._dimension = 384

    @property
    def dimension(self):
        return self._dimension

    def embed(self, text: str):
        if self.model is None:
            raise RuntimeError("Embedder not loaded")
        if isinstance(self.model, FallbackEmbedder):
            return self.model.embed(text)
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].astype(np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 32):
        if self.model is None:
            raise RuntimeError("Embedder not loaded")
        if isinstance(self.model, FallbackEmbedder):
            return np.array(
                [self.model.embed(t) for t in texts],
                dtype=np.float32
            )
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.astype(np.float32)


class FallbackEmbedder:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._vocab = {}
        self._projection = None
        self._rng = np.random.RandomState(42)

    def _get_projection(self, vocab_size):
        if (self._projection is None
                or self._projection.shape[0] != vocab_size):
            self._projection = self._rng.randn(
                vocab_size, self.dimension
            ).astype(np.float32)
            norms = np.linalg.norm(
                self._projection, axis=0, keepdims=True
            )
            self._projection /= np.clip(norms, 1e-8, None)
        return self._projection

    def _tokenize(self, text):
        import re
        return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())

    def embed(self, text):
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
        vocab_size = len(self._vocab)
        bow = np.zeros(vocab_size, dtype=np.float32)
        for token in tokens:
            bow[self._vocab[token]] += 1
        bow /= (np.sum(bow) + 1e-8)
        proj = self._get_projection(vocab_size)
        embedding = bow @ proj
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding /= norm
        return embedding.astype(np.float32)