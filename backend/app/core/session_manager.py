"""
Session Manager - Handles per-session conversation history.

Stores conversation history in memory (with optional Redis backing).
"""

import time
import logging
from typing import Optional
from collections import defaultdict

from app.core.config import settings

logger = logging.getLogger(__name__)


class SessionManager:
    """
    In-memory session manager with optional Redis persistence.
    
    Each session stores a list of {"role": ..., "content": ...} dicts.
    Sessions expire after 30 minutes of inactivity.
    """

    SESSION_TTL = 30 * 60  # 30 minutes

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._redis = None
        self._try_init_redis()

    def _try_init_redis(self):
        """Try to connect to Redis if configured."""
        if not settings.REDIS_URL:
            return
        try:
            import redis
            self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
            self._redis.ping()
            logger.info("✅ Redis connected for session storage")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory sessions: {e}")
            self._redis = None

    def get_history(self, session_id: str) -> list[dict]:
        """Get conversation history for a session."""
        self._cleanup_expired()

        if self._redis:
            return self._get_from_redis(session_id)

        session = self._sessions.get(session_id)
        if not session:
            return []
        session["last_active"] = time.time()
        return session["history"]

    def add_turn(self, session_id: str, user_message: str, assistant_message: str):
        """Add a conversation turn to session history."""
        if self._redis:
            self._add_to_redis(session_id, user_message, assistant_message)
            return

        if session_id not in self._sessions:
            self._sessions[session_id] = {"history": [], "last_active": time.time()}

        session = self._sessions[session_id]
        session["history"].extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ])
        session["last_active"] = time.time()

        # Trim to max turns
        max_msgs = settings.MAX_HISTORY_TURNS * 2
        if len(session["history"]) > max_msgs:
            session["history"] = session["history"][-max_msgs:]

    def clear_session(self, session_id: str):
        """Clear a session's history."""
        if self._redis:
            self._redis.delete(f"session:{session_id}")
        self._sessions.pop(session_id, None)

    def _cleanup_expired(self):
        """Remove expired sessions from memory."""
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s["last_active"] > self.SESSION_TTL
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired sessions")

    def _get_from_redis(self, session_id: str) -> list[dict]:
        """Load session history from Redis."""
        import json
        data = self._redis.get(f"session:{session_id}")
        if not data:
            return []
        return json.loads(data)

    def _add_to_redis(self, session_id: str, user_msg: str, assistant_msg: str):
        """Save session turn to Redis."""
        import json
        key = f"session:{session_id}"
        history = self._get_from_redis(session_id)
        history.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ])
        max_msgs = settings.MAX_HISTORY_TURNS * 2
        if len(history) > max_msgs:
            history = history[-max_msgs:]
        self._redis.setex(key, self.SESSION_TTL, json.dumps(history))


# Global singleton
session_manager = SessionManager()
