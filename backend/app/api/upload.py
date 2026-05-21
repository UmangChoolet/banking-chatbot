"""
Upload API Endpoint

POST /api/upload - Upload and ingest a document
GET /api/documents - List ingested documents
"""

import os
import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Track ingested documents in memory
_ingested_documents: list[dict] = []


class UploadResponse(BaseModel):
    message: str
    file_name: str
    chunks_created: int
    total_vectors: int


@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, DOCX) and ingest it into the vector store.
    
    Process:
    1. Validate file type and size
    2. Save to disk
    3. Extract text and chunk
    4. Generate embeddings
    5. Store in FAISS vector DB
    """
    rag = request.app.state.rag
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready")

    # Validate file type
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Maximum: {settings.MAX_FILE_SIZE_MB}MB",
        )

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Save file to disk
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    save_path = Path(settings.UPLOAD_DIR) / safe_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(content)

    logger.info(f"Saved uploaded file: {save_path}")

    # Ingest into RAG pipeline
    try:
        result = await rag.ingest_file(str(save_path))

        # Track document
        doc_record = {
            "original_name": file.filename,
            "stored_name": safe_name,
            "size_mb": round(size_mb, 2),
            "chunks": result["chunks_created"],
            "file_type": file_ext,
        }
        _ingested_documents.append(doc_record)

        return UploadResponse(
            message=f"Successfully ingested '{file.filename}'",
            file_name=file.filename,
            chunks_created=result["chunks_created"],
            total_vectors=result["total_vectors"],
        )

    except ValueError as e:
        # Clean up saved file
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/documents")
async def list_documents(request: Request):
    """List all ingested documents and vector store stats."""
    rag = request.app.state.rag
    total_vectors = 0
    if rag and rag.vector_store:
        total_vectors = rag.vector_store.count()

    return {
        "documents": _ingested_documents,
        "total_documents": len(_ingested_documents),
        "total_vectors": total_vectors,
    }
