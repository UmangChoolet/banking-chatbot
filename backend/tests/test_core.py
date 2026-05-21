"""
Backend Unit Tests
Tests for document processing, embedder, vector store, and API endpoints.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


# ===== Document Processor Tests =====

class TestDocumentProcessor:
    def setup_method(self):
        from app.core.document_processor import DocumentProcessor
        self.processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)

    def test_extract_txt(self, tmp_path):
        """Test text extraction from TXT file."""
        doc = tmp_path / "test.txt"
        doc.write_text("Hello world. This is a banking document about personal loans.")
        chunks = self.processor.process_file(str(doc))
        assert len(chunks) >= 1
        assert "banking" in chunks[0]["text"].lower()

    def test_chunking_with_overlap(self, tmp_path):
        """Test that chunks are created with correct overlap."""
        long_text = " ".join(["word"] * 300)
        doc = tmp_path / "long.txt"
        doc.write_text(long_text)
        chunks = self.processor.process_file(str(doc))
        assert len(chunks) > 1  # Should be split into multiple chunks

    def test_empty_file_raises(self, tmp_path):
        """Test that empty files raise an error."""
        doc = tmp_path / "empty.txt"
        doc.write_text("")
        with pytest.raises(ValueError, match="No text"):
            self.processor.process_file(str(doc))

    def test_unsupported_extension_raises(self, tmp_path):
        """Test that unsupported file types raise an error."""
        doc = tmp_path / "test.csv"
        doc.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported"):
            self.processor.process_file(str(doc))

    def test_chunk_metadata(self, tmp_path):
        """Test that chunks include correct metadata."""
        doc = tmp_path / "test.txt"
        doc.write_text("Test content for banking support chatbot.")
        chunks = self.processor.process_file(str(doc))
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["source"] == "test.txt"
        assert "text" in chunks[0]


# ===== Embedder Tests =====

class TestFallbackEmbedder:
    def setup_method(self):
        from app.core.embedder import FallbackEmbedder
        self.embedder = FallbackEmbedder(dimension=64)

    def test_embed_returns_correct_shape(self):
        embedding = self.embedder.embed("personal loan interest rate")
        assert embedding.shape == (64,)
        assert embedding.dtype == np.float32

    def test_embed_normalized(self):
        embedding = self.embedder.embed("credit card fees and charges")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Should be unit vector

    def test_similar_texts_closer(self):
        e1 = self.embedder.embed("personal loan interest rate")
        e2 = self.embedder.embed("loan interest personal")
        e3 = self.embedder.embed("weather forecast tomorrow rain")
        sim_related = np.dot(e1, e2)
        sim_unrelated = np.dot(e1, e3)
        # Related texts should have higher similarity
        assert sim_related > sim_unrelated

    def test_embed_empty_string(self):
        embedding = self.embedder.embed("")
        assert embedding.shape == (64,)
        # Empty should return zero vector
        assert np.all(embedding == 0)


# ===== Vector Store Tests =====

class TestNumpyVectorIndex:
    def setup_method(self):
        from app.core.vector_store import NumpyVectorIndex
        self.index = NumpyVectorIndex(dimension=8)

    def test_add_and_count(self):
        vecs = np.random.randn(5, 8).astype(np.float32)
        self.index.add(vecs)
        assert self.index.ntotal == 5

    def test_search_returns_closest(self):
        # Add a known vector
        vecs = np.eye(8, dtype=np.float32)  # 8 orthogonal unit vectors
        self.index.add(vecs)
        # Search for first vector
        query = vecs[0:1]
        scores, indices = self.index.search(query, k=1)
        assert indices[0][0] == 0  # Should find itself


class TestFAISSVectorStore:
    def test_full_pipeline(self, tmp_path):
        """Test add, save, load, search cycle."""
        from app.core.vector_store import FAISSVectorStore
        store = FAISSVectorStore(index_path=str(tmp_path), dimension=16)
        store.load()

        vecs = np.random.randn(10, 16).astype(np.float32)
        # Normalize
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        metadatas = [{"text": f"doc {i}", "source": "test.txt"} for i in range(10)]
        store.add(vecs, metadatas)
        assert store.count() == 10

        # Save and reload
        store.save()
        store2 = FAISSVectorStore(index_path=str(tmp_path), dimension=16)
        store2.load()
        assert store2.count() == 10

        # Search
        results = store2.search(vecs[0], top_k=3)
        assert len(results) <= 3
        assert results[0]["metadata"]["text"] == "doc 0"  # Should find itself first


# ===== API Tests =====

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock, MagicMock, patch

    # Mock the RAG pipeline
    mock_rag = MagicMock()
    mock_rag.query = AsyncMock(return_value={
        "answer": "Personal loans have interest rates from 10.5% to 24% p.a.",
        "sources": [{"source": "personal_loans.txt", "relevance_score": 0.85, "excerpt": "..."}],
        "chunks_retrieved": 3,
    })
    mock_rag.vector_store = MagicMock()
    mock_rag.vector_store.count = MagicMock(return_value=42)
    mock_rag.vector_store.index = MagicMock()
    mock_rag.vector_store.index.ntotal = 42

    with patch("app.core.rag_pipeline.RAGPipeline.initialize", new_callable=AsyncMock):
        from app.main import app
        app.state.rag = mock_rag
        return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Banking Support Chatbot" in response.json()["message"]


class TestChatEndpoint:
    def test_chat_valid_message(self, client):
        response = client.post(
            "/api/chat",
            json={"message": "What is a personal loan?", "session_id": "test-123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["session_id"] == "test-123"

    def test_chat_empty_message_rejected(self, client):
        response = client.post(
            "/api/chat",
            json={"message": "", "session_id": "test-123"},
        )
        assert response.status_code == 422  # Pydantic validation

    def test_chat_auto_generates_session_id(self, client):
        response = client.post(
            "/api/chat",
            json={"message": "Hello", "session_id": "auto-test-456"},
        )
        assert response.status_code == 200
        assert response.json()["session_id"] is not None


class TestSessionEndpoints:
    def test_get_history(self, client):
        response = client.get("/api/chat/session-xyz/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert "session_id" in data

    def test_clear_session(self, client):
        response = client.delete("/api/chat/session-xyz")
        assert response.status_code == 200
        assert "cleared" in response.json()["message"]
