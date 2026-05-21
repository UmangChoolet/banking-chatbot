"""
Banking Support Chatbot - FastAPI Backend
RAG-powered chatbot with FAISS vector database
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os

from app.api.chat import router as chat_router
from app.api.upload import router as upload_router
from app.core.rag_pipeline import RAGPipeline
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on startup."""
    logger.info("🚀 Starting Banking Chatbot Backend...")
    rag = RAGPipeline()
    await rag.initialize()
    app.state.rag = rag
    logger.info("✅ RAG pipeline initialized successfully")
    yield
    logger.info("🛑 Shutting down...")


app = FastAPI(
    title="Banking Support Chatbot API",
    description="RAG-powered AI assistant for banking customer support",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(upload_router, prefix="/api", tags=["Upload"])


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    rag = getattr(app.state, "rag", None)
    vector_count = 0
    if rag and rag.vector_store:
        try:
            vector_count = rag.vector_store.index.ntotal
        except Exception:
            pass
    return {
        "status": "healthy",
        "version": "1.0.0",
        "vector_db_documents": vector_count,
        "model": settings.EMBEDDING_MODEL,
        "llm": settings.LLM_MODEL,
    }


@app.get("/", tags=["Root"])
async def root():
    return {"message": "Banking Support Chatbot API", "docs": "/docs"}
