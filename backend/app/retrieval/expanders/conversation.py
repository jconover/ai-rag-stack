"""Conversation-aware query expander.

Wraps the existing ConversationContextExpander to conform to the QueryExpander
interface. Resolves pronouns and references using conversation history.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple

from app.retrieval.expanders.base import BaseExpander, ExpansionResult
from app.config import settings

logger = logging.getLogger(__name__)


class ConversationExpander(BaseExpander):
    """Conversation-aware query expander.

    Improves retrieval for follow-up questions by incorporating context
    from recent conversation history.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._expander = None

    def _get_expander(self):
        """Lazy load to avoid circular imports."""
        if self._expander is None:
            from app.conversation_context import conversation_expander
            self._expander = conversation_expander
        return self._expander

    @property
    def name(self) -> str:
        return "conversation"

    def should_expand(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Check if conversation context should be applied."""
        return self._get_expander().should_expand(query, conversation_history)

    def expand(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ExpansionResult:
        """Expand the query using conversation context synchronously."""
        conv_result = self._get_expander().expand_query(query, conversation_history)

        return ExpansionResult(
            original_query=conv_result.original_query,
            expanded_query=conv_result.expanded_query,
            expanded=conv_result.expanded,
            skip_reason=conv_result.skip_reason,
            context_terms=conv_result.context_terms,
            metadata={
                "resolved_references": conv_result.resolved_references,
            },
            expansion_time_ms=0.0,
            expander_name=self.name,
            error=None,
        )

    async def expand_async(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ExpansionResult:
        """Expand the query using conversation context asynchronously."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.expand(query, conversation_history=conversation_history)
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            **super().get_config(),
            "enabled": getattr(settings, 'conversation_context_enabled', True),
            "history_limit": getattr(settings, 'conversation_context_history_limit', 3),
            "max_terms": getattr(settings, 'conversation_context_max_terms', 10),
        }

    def is_available(self) -> bool:
        return getattr(settings, 'conversation_context_enabled', True)


__all__ = ["ConversationExpander"]
