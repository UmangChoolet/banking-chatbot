"""
Chat API Endpoint

POST /api/chat - Standard chat
POST /api/chat/stream - Streaming chat (SSE)
DELETE /api/chat/{session_id} - Clear session
"""

import uuid
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.session_manager import session_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Session identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is a personal loan and what are the interest rates?",
                "session_id": "user-session-123",
            }
        }


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[dict] = []
    chunks_retrieved: int = 0


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """
    Standard (non-streaming) chat endpoint.
    
    - Retrieves conversation history for the session
    - Runs RAG pipeline (embed → retrieve → generate)
    - Stores the turn in session memory
    - Returns response with source attribution
    """
    rag = request.app.state.rag
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready")

    # Sanitize input
    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = body.session_id or str(uuid.uuid4())

    try:
        # Get conversation history
        history = session_manager.get_history(session_id)

        # Run RAG pipeline
        result = await rag.query(
            user_message=user_message,
            conversation_history=history,
            session_id=session_id,
        )

        # Save turn to session
        session_manager.add_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=result["answer"],
        )

        return ChatResponse(
            answer=result["answer"],
            session_id=session_id,
            sources=result.get("sources", []),
            chunks_retrieved=result.get("chunks_retrieved", 0),
        )

    except Exception as e:
        logger.error(f"Chat error for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events.
    Yields text chunks as they are generated.
    """
    rag = request.app.state.rag
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready")

    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = body.session_id or str(uuid.uuid4())
    history = session_manager.get_history(session_id)

    async def event_generator():
        full_response = ""
        try:
            # Get retrieval context first
            query_embedding = rag.embedder.embed(user_message)
            retrieved_chunks = rag.vector_store.search(
                query_embedding,
                top_k=5,
                threshold=0.3,
            )
            context = rag._build_context(retrieved_chunks)
            messages = rag._build_messages(user_message, context, history)

            # Stream LLM response
            async for chunk in rag.llm.generate_stream(messages):
                full_response += chunk
                yield f"data: {chunk}\n\n"

            # Save complete turn
            session_manager.add_turn(session_id, user_message, full_response)
            yield f"data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-ID": session_id,
        },
    )


@router.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    session_manager.clear_session(session_id)
    return {"message": f"Session {session_id} cleared", "session_id": session_id}


@router.get("/chat/{session_id}/history")
async def get_history(session_id: str):
    """Get conversation history for a session."""
    history = session_manager.get_history(session_id)
    return {"session_id": session_id, "history": history, "turns": len(history) // 2}
