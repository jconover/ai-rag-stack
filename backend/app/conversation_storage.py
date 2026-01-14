"""Tiered conversation storage with automatic summarization.

Implements a three-tier storage system:
- Tier 1: Recent messages in Redis (fast access, 24h TTL)
- Tier 2: Conversation summaries in Redis (7d TTL, smaller footprint)
- Tier 3: Full history in PostgreSQL (for valuable/starred conversations)

The summarization uses the configured Ollama model to generate concise
summaries of older messages, allowing long conversations to maintain
context while keeping memory usage low.
"""

import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
import httpx

from app.config import settings
from app.redis_client import get_redis_string_client

logger = logging.getLogger(__name__)


class ConversationStorage:
    """Manages tiered conversation storage with summarization.

    This class provides intelligent conversation storage that automatically
    summarizes older messages to maintain context while reducing storage
    requirements. It uses Redis for fast access to recent messages and
    summaries.

    Configuration is read from settings at initialization time, allowing
    environment-based customization via:
    - CONVERSATION_SUMMARY_THRESHOLD: Messages before summarizing (default: 10)
    - CONVERSATION_SUMMARY_TTL: Summary TTL in seconds (default: 7 days)
    - CONVERSATION_RECENT_TTL: Recent messages TTL in seconds (default: 24h)
    - CONVERSATION_RECENT_TO_KEEP: Messages to keep after summarizing (default: 5)

    Attributes:
        redis: Redis client for storage operations
        summary_threshold: Number of messages before triggering summarization
        summary_ttl: TTL for summary keys in seconds
        recent_ttl: TTL for recent messages in seconds
        recent_to_keep: Number of recent messages to keep after summarization
    """

    def __init__(
        self,
        summary_threshold: Optional[int] = None,
        summary_ttl: Optional[int] = None,
        recent_ttl: Optional[int] = None,
        recent_to_keep: Optional[int] = None,
    ):
        """Initialize the conversation storage.

        Args:
            summary_threshold: Number of messages before triggering summarization.
                              Defaults to settings.conversation_summary_threshold.
            summary_ttl: TTL for summary keys in seconds.
                        Defaults to settings.conversation_summary_ttl.
            recent_ttl: TTL for recent messages in seconds.
                       Defaults to settings.conversation_recent_ttl.
            recent_to_keep: Number of recent messages to keep after summarization.
                           Defaults to settings.conversation_recent_to_keep.
        """
        self.redis = get_redis_string_client()
        self.summary_threshold = summary_threshold or settings.conversation_summary_threshold
        self.summary_ttl = summary_ttl or settings.conversation_summary_ttl
        self.recent_ttl = recent_ttl or settings.conversation_recent_ttl
        self.recent_to_keep = recent_to_keep or settings.conversation_recent_to_keep

    def _messages_key(self, session_id: str) -> str:
        """Generate Redis key for storing messages.

        Args:
            session_id: The session identifier

        Returns:
            Redis key string for messages list
        """
        return f"conversation:{session_id}:messages"

    def _summary_key(self, session_id: str) -> str:
        """Generate Redis key for storing summary.

        Args:
            session_id: The session identifier

        Returns:
            Redis key string for summary
        """
        return f"conversation:{session_id}:summary"

    def _metadata_key(self, session_id: str) -> str:
        """Generate Redis key for storing conversation metadata.

        Args:
            session_id: The session identifier

        Returns:
            Redis key string for metadata
        """
        return f"conversation:{session_id}:metadata"

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message and trigger summarization if threshold reached.

        This method stores the message with timestamp metadata and checks
        if the conversation has grown large enough to warrant summarization.

        Args:
            session_id: The session identifier
            role: Message role ('user' or 'assistant')
            content: The message content
        """
        key = self._messages_key(session_id)
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()
        pipe.rpush(key, message)
        pipe.expire(key, self.recent_ttl)
        pipe.execute()

        # Check if we should summarize
        message_count = self.redis.llen(key)
        if message_count >= self.summary_threshold:
            await self._summarize_and_compact(session_id)

    def add_message_sync(self, session_id: str, role: str, content: str) -> None:
        """Synchronous version of add_message for backward compatibility.

        This method stores the message without triggering summarization.
        Summarization should be triggered separately via check_and_summarize().

        Args:
            session_id: The session identifier
            role: Message role ('user' or 'assistant')
            content: The message content
        """
        key = self._messages_key(session_id)
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })

        pipe = self.redis.pipeline()
        pipe.rpush(key, message)
        pipe.expire(key, self.recent_ttl)
        pipe.execute()

    async def check_and_summarize(self, session_id: str) -> bool:
        """Check if summarization is needed and perform it.

        This is a separate method to allow decoupled summarization from
        message storage, useful for async workflows.

        Args:
            session_id: The session identifier

        Returns:
            True if summarization was performed, False otherwise
        """
        key = self._messages_key(session_id)
        message_count = self.redis.llen(key)

        if message_count >= self.summary_threshold:
            await self._summarize_and_compact(session_id)
            return True
        return False

    async def get_context(self, session_id: str, max_recent: int = 5) -> Dict:
        """Get conversation context including summary and recent messages.

        This method retrieves the full context needed for RAG queries,
        including any accumulated summaries and recent messages.

        Args:
            session_id: The session identifier
            max_recent: Maximum number of recent messages to return

        Returns:
            Dictionary with:
                - summary: Accumulated conversation summary (or None)
                - recent_messages: List of recent message dicts
                - has_history: Boolean indicating if any history exists
                - message_count: Total messages in current window
        """
        summary = self.redis.get(self._summary_key(session_id))
        recent_raw = self.redis.lrange(self._messages_key(session_id), -max_recent, -1)

        recent = [json.loads(m) for m in recent_raw] if recent_raw else []

        return {
            "summary": summary if summary else None,
            "recent_messages": recent,
            "has_history": summary is not None or len(recent) > 0,
            "message_count": self.redis.llen(self._messages_key(session_id))
        }

    def get_context_sync(self, session_id: str, max_recent: int = 5) -> Dict:
        """Synchronous version of get_context.

        Args:
            session_id: The session identifier
            max_recent: Maximum number of recent messages to return

        Returns:
            Dictionary with summary, recent_messages, has_history, message_count
        """
        summary = self.redis.get(self._summary_key(session_id))
        recent_raw = self.redis.lrange(self._messages_key(session_id), -max_recent, -1)

        recent = [json.loads(m) for m in recent_raw] if recent_raw else []

        return {
            "summary": summary if summary else None,
            "recent_messages": recent,
            "has_history": summary is not None or len(recent) > 0,
            "message_count": self.redis.llen(self._messages_key(session_id))
        }

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history in the legacy format.

        This method provides backward compatibility with the existing
        get_conversation_history() function.

        Args:
            session_id: The session identifier
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries with role and content
        """
        recent_raw = self.redis.lrange(self._messages_key(session_id), -limit, -1)
        return [json.loads(m) for m in recent_raw] if recent_raw else []

    async def _summarize_and_compact(self, session_id: str) -> None:
        """Summarize older messages and keep only recent ones.

        This method:
        1. Retrieves messages older than the recent window
        2. Generates a summary using the configured LLM
        3. Appends to any existing summary
        4. Trims the message list to keep only recent messages

        Args:
            session_id: The session identifier
        """
        key = self._messages_key(session_id)

        # Get all messages except the most recent ones we want to keep
        messages_raw = self.redis.lrange(key, 0, -(self.recent_to_keep + 1))

        if not messages_raw or len(messages_raw) < self.recent_to_keep:
            return

        messages = [json.loads(m) for m in messages_raw]

        # Generate summary using LLM
        summary_prompt = self._build_summary_prompt(messages)

        try:
            # Use httpx for async HTTP calls to Ollama
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{settings.ollama_host}/api/chat",
                    json={
                        "model": settings.ollama_default_model,
                        "messages": [{"role": "user", "content": summary_prompt}],
                        "options": {
                            "num_predict": 200,
                            "temperature": 0.3
                        },
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                summary = result.get("message", {}).get("content", "")

            if not summary:
                logger.warning(f"Empty summary generated for session {session_id}")
                return

            # Append to existing summary if present
            existing_summary = self.redis.get(self._summary_key(session_id))
            if existing_summary:
                # Combine summaries with a separator
                summary = f"{existing_summary}\n\n[Continued]: {summary}"

            # Store summary with extended TTL
            self.redis.setex(self._summary_key(session_id), self.summary_ttl, summary)

            # Trim messages to keep only recent ones
            # LTRIM keeps elements from start to end (inclusive)
            self.redis.ltrim(key, -self.recent_to_keep, -1)

            logger.info(
                f"Summarized {len(messages)} messages for session {session_id}. "
                f"Summary length: {len(summary)} chars"
            )

        except httpx.HTTPError as e:
            logger.warning(f"Failed to summarize conversation (HTTP error): {e}")
        except Exception as e:
            logger.warning(f"Failed to summarize conversation: {e}")

    def _build_summary_prompt(self, messages: List[Dict]) -> str:
        """Build a prompt for summarizing conversation messages.

        Args:
            messages: List of message dictionaries to summarize

        Returns:
            Prompt string for the LLM
        """
        # Format conversation with truncation for very long messages
        conversation_parts = []
        for m in messages:
            role = m.get('role', 'unknown').upper()
            content = m.get('content', '')[:500]  # Truncate long messages
            if len(m.get('content', '')) > 500:
                content += "..."
            conversation_parts.append(f"{role}: {content}")

        conversation = "\n".join(conversation_parts)

        return f"""Summarize this conversation in 2-3 sentences, focusing on:
- The main topics discussed
- Any decisions or solutions reached
- Key technical details mentioned

Be concise and preserve important context that would be useful for continuing the conversation.

Conversation:
{conversation}

Summary:"""

    def clear_session(self, session_id: str) -> None:
        """Clear all data for a session.

        Args:
            session_id: The session identifier
        """
        pipe = self.redis.pipeline()
        pipe.delete(self._messages_key(session_id))
        pipe.delete(self._summary_key(session_id))
        pipe.delete(self._metadata_key(session_id))
        pipe.execute()

        logger.info(f"Cleared all data for session {session_id}")

    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics about a conversation session.

        Args:
            session_id: The session identifier

        Returns:
            Dictionary with session statistics
        """
        messages_key = self._messages_key(session_id)
        summary_key = self._summary_key(session_id)

        message_count = self.redis.llen(messages_key)
        messages_ttl = self.redis.ttl(messages_key)

        summary = self.redis.get(summary_key)
        summary_ttl = self.redis.ttl(summary_key)

        return {
            "session_id": session_id,
            "message_count": message_count,
            "messages_ttl_seconds": messages_ttl if messages_ttl > 0 else None,
            "has_summary": summary is not None,
            "summary_length": len(summary) if summary else 0,
            "summary_ttl_seconds": summary_ttl if summary_ttl > 0 else None,
        }


# Singleton instance for use across the application
conversation_storage = ConversationStorage()
