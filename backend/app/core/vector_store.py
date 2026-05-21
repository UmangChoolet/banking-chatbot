"""
FAISS Vector Store - Persistent vector database using FAISS.

Features:
- L2 + inner product similarity search
- Persistent index (save/load from disk)
- Metadata storage alongside vectors
- Similarity threshold filtering
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store with persistent storage.
    
    Uses IndexFlatIP (Inner Product / cosine similarity on normalized vectors).
    Metadata (text, source) is stored in a parallel list and pickled.
    """

    def __init__(self, index_path: str, dimension: int = 384):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index = None
        self.metadatas: list[dict] = []
        self._faiss_available = False

    def _init_index(self):
        """Initialize a new FAISS index."""
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)
            self._faiss_available = True
            logger.info(f"FAISS index initialized (dim={self.dimension})")
        except ImportError:
            logger.warning(
                "FAISS not installed. Using NumPy fallback vector store. "
                "Install with: pip install faiss-cpu"
            )
            self.index = NumpyVectorIndex(self.dimension)
            self._faiss_available = False

    def load(self):
        """Load existing index and metadata from disk, or create new."""
        index_file = self.index_path / "index.faiss"
        meta_file = self.index_path / "metadata.pkl"

        if index_file.exists() and meta_file.exists():
            try:
                if self._try_load_faiss(index_file, meta_file):
                    logger.info(f"✅ Loaded FAISS index with {self.count()} vectors")
                    return
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new.")

        # Create fresh index
        self._init_index()
        self.metadatas = []
        logger.info("Created new vector store")

    def _try_load_faiss(self, index_file: Path, meta_file: Path) -> bool:
        """Attempt to load FAISS index from disk."""
        try:
            import faiss
            self.index = faiss.read_index(str(index_file))
            self._faiss_available = True
        except ImportError:
            self.index = NumpyVectorIndex(self.dimension)
            if (self.index_path / "numpy_store.pkl").exists():
                with open(self.index_path / "numpy_store.pkl", "rb") as f:
                    self.index = pickle.load(f)

        with open(meta_file, "rb") as f:
            self.metadatas = pickle.load(f)
        return True

    def save(self):
        """Persist the index and metadata to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        meta_file = self.index_path / "metadata.pkl"

        try:
            import faiss
            index_file = self.index_path / "index.faiss"
            faiss.write_index(self.index, str(index_file))
        except (ImportError, Exception):
            # Save numpy index
            numpy_file = self.index_path / "numpy_store.pkl"
            with open(numpy_file, "wb") as f:
                pickle.dump(self.index, f)

        with open(meta_file, "wb") as f:
            pickle.dump(self.metadatas, f)

        logger.info(f"Saved vector store with {self.count()} vectors")

    def add(self, embeddings: np.ndarray, metadatas: list[dict]):
        """Add vectors and their metadata to the store."""
        if self.index is None:
            self._init_index()

        embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-8, None)

        self.index.add(embeddings)
        self.metadatas.extend(metadatas)

        logger.info(f"Added {len(metadatas)} vectors. Total: {self.count()}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """
        Semantic similarity search.
        Returns list of {score, metadata} dicts sorted by score descending.
        """
        if self.index is None or self.count() == 0:
            return []

        # Normalize query
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query /= norm

        k = min(top_k, self.count())
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            if float(score) < threshold:
                continue
            results.append({
                "score": float(score),
                "metadata": self.metadatas[idx],
            })

        return results

    def count(self) -> int:
        """Return number of vectors in the store."""
        if self.index is None:
            return 0
        try:
            return self.index.ntotal
        except AttributeError:
            return len(getattr(self.index, "vectors", []))

    def is_empty(self) -> bool:
        return self.count() == 0


class NumpyVectorIndex:
    """
    Pure-NumPy fallback when FAISS is not available.
    O(n) brute-force search - fine for small corpora (<10k vectors).
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: list[np.ndarray] = []
        self.ntotal = 0

    def add(self, embeddings: np.ndarray):
        for vec in embeddings:
            self.vectors.append(vec.astype(np.float32))
        self.ntotal = len(self.vectors)

    def search(self, query: np.ndarray, k: int):
        """Brute-force inner product search."""
        if not self.vectors:
            return np.array([[]], dtype=np.float32), np.array([[-1]], dtype=np.int64)

        matrix = np.array(self.vectors, dtype=np.float32)
        scores = matrix @ query.T  # (n, 1)
        scores = scores.flatten()

        k = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]
        top_scores = scores[top_indices]

        return top_scores.reshape(1, -1), top_indices.reshape(1, -1).astype(np.int64)
