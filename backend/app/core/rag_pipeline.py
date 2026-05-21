"""
RAG Pipeline - Core retrieval-augmented generation logic.

Flow:
  Document → Chunking → Embedding → FAISS Index
  Query → Embed → Similarity Search → Context → LLM → Response
"""

import os
import pickle
import logging
import asyncio
from pathlib import Path
from typing import Optional
import numpy as np

from app.core.config import settings
from app.core.document_processor import DocumentProcessor
from app.core.embedder import Embedder
from app.core.vector_store import FAISSVectorStore
from app.core.llm_client import LLMClient

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline:
      - Document ingestion + chunking
      - Embedding generation (sentence-transformers, free)
      - FAISS vector storage
      - Semantic retrieval
      - LLM response generation with context
    """

    def __init__(self):
        self.doc_processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.embedder = Embedder(model_name=settings.EMBEDDING_MODEL)
        self.vector_store = FAISSVectorStore(
            index_path=settings.VECTOR_DB_PATH,
            dimension=384,  # all-MiniLM-L6-v2 output dim
        )
        self.llm = LLMClient()
        self._initialized = False

    async def initialize(self):
        """Load models and existing vector index."""
        logger.info("Loading embedding model...")
        self.embedder.load()

        logger.info("Loading FAISS vector store...")
        self.vector_store.load()

        # Ingest sample data if vector store is empty
        if self.vector_store.is_empty():
            logger.info("Vector store is empty. Ingesting sample banking documents...")
            await self._ingest_sample_data()

        self._initialized = True
        logger.info(f"✅ RAG pipeline ready. Vectors in store: {self.vector_store.count()}")

    async def _ingest_sample_data(self):
        """Ingest built-in sample banking documents."""
        sample_dir = Path(__file__).parent.parent.parent / "data" / "sample_docs"
        if sample_dir.exists():
            for doc_path in sample_dir.glob("*.txt"):
                try:
                    await self.ingest_file(str(doc_path))
                    logger.info(f"Ingested sample doc: {doc_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to ingest {doc_path.name}: {e}")

    async def ingest_file(self, file_path: str) -> dict:
        """
        Ingest a document file into the vector store.
        Returns metadata about the ingestion.
        """
        logger.info(f"Ingesting file: {file_path}")

        # 1. Load and chunk document
        chunks = self.doc_processor.process_file(file_path)
        if not chunks:
            raise ValueError(f"No text extracted from {file_path}")

        logger.info(f"Extracted {len(chunks)} chunks from {file_path}")

        # 2. Generate embeddings
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        # 3. Store in FAISS
        metadatas = [
            {
                "source": os.path.basename(file_path),
                "chunk_index": c["chunk_index"],
                "text": c["text"],
            }
            for c in chunks
        ]
        self.vector_store.add(embeddings, metadatas)
        self.vector_store.save()

        return {
            "file": os.path.basename(file_path),
            "chunks_created": len(chunks),
            "total_vectors": self.vector_store.count(),
        }

    async def query(
        self,
        user_message: str,
        conversation_history: list[dict],
        session_id: str,
    ) -> dict:
        """
        Full RAG query:
          1. Embed user query
          2. Retrieve top-K relevant chunks
          3. Build context-aware prompt
          4. Generate LLM response
        """
        if not self._initialized:
            raise RuntimeError("RAG pipeline not initialized")

        # 1. Embed the query
        query_embedding = self.embedder.embed(user_message)

        # 2. Semantic retrieval from FAISS
        retrieved_chunks = self.vector_store.search(
            query_embedding,
            top_k=settings.TOP_K_RETRIEVAL,
            threshold=settings.SIMILARITY_THRESHOLD,
        )

        # 3. Build context string
        context = self._build_context(retrieved_chunks)

        # 4. Build prompt with conversation history
        messages = self._build_messages(
            user_message=user_message,
            context=context,
            conversation_history=conversation_history,
        )

        # 5. Generate response from LLM
        response_text = await self.llm.generate(messages)

        return {
            "answer": response_text,
            "sources": [
                {
                    "source": c["metadata"]["source"],
                    "relevance_score": round(float(c["score"]), 3),
                    "excerpt": c["metadata"]["text"][:200] + "..."
                    if len(c["metadata"]["text"]) > 200
                    else c["metadata"]["text"],
                }
                for c in retrieved_chunks
            ],
            "chunks_retrieved": len(retrieved_chunks),
        }

    def _build_context(self, retrieved_chunks: list[dict]) -> str:
        """Format retrieved chunks into context string."""
        if not retrieved_chunks:
            return "No specific banking documents found for this query."

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk["metadata"].get("source", "Unknown")
            text = chunk["metadata"].get("text", "")
            score = chunk.get("score", 0)
            context_parts.append(
                f"[Document {i} | Source: {source} | Relevance: {score:.2f}]\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _build_messages(
        self,
        user_message: str,
        context: str,
        conversation_history: list[dict],
    ) -> list[dict]:
        """Build the full message list for the LLM."""
        system_prompt = f"""You are a helpful and knowledgeable banking support assistant for a modern fintech company. Your role is to assist customers with banking-related queries including loans, credit cards, account management, and general banking policies.

RETRIEVED BANKING KNOWLEDGE:
{context}

INSTRUCTIONS:
- Answer based primarily on the retrieved documents above
- If the documents contain relevant information, cite it clearly
- If the question is outside the retrieved context, use your general banking knowledge but indicate this
- Always be accurate, professional, and clear
- For sensitive queries (fraud, account issues), advise contacting the bank directly
- Maintain conversation context and remember what was discussed earlier in this session
- Keep responses concise but complete
- Format numbers, rates, and policies clearly"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add recent conversation history (last N turns)
        history_turns = conversation_history[-(settings.MAX_HISTORY_TURNS * 2):]
        for msg in history_turns:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages
