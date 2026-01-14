"""HyDE (Hypothetical Document Embeddings) query expander.

Wraps the existing HyDEExpander from app.query_expansion to conform to the
QueryExpander interface.
"""

import logging
from typing import Optional, Dict, Any, Tuple

from app.retrieval.expanders.base import BaseExpander, ExpansionResult
from app.config import settings

logger = logging.getLogger(__name__)


class HyDEExpander(BaseExpander):
    """Hypothetical Document Embeddings query expander.

    Generates a hypothetical document that would answer the query,
    then uses that document's embedding for similarity search.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._expander = None

    def _get_expander(self):
        """Lazy load the HyDE expander to avoid circular imports."""
        if self._expander is None:
            from app.query_expansion import hyde_expander
            self._expander = hyde_expander
        return self._expander

    @property
    def name(self) -> str:
        return "hyde"

    def should_expand(self, query: str, **kwargs) -> Tuple[bool, Optional[str]]:
        """Check if HyDE should be applied to this query."""
        return self._get_expander().should_expand(query)

    def expand(self, query: str, **kwargs) -> ExpansionResult:
        """Expand the query using HyDE synchronously."""
        hyde_result = self._get_expander().expand_sync(query)

        return ExpansionResult(
            original_query=hyde_result.original_query,
            expanded_query=hyde_result.hypothetical_document or hyde_result.original_query,
            expanded=hyde_result.expanded,
            skip_reason=hyde_result.skip_reason or hyde_result.error,
            context_terms=[],
            metadata={
                "model_used": hyde_result.model_used,
                "hypothetical_document": hyde_result.hypothetical_document,
            },
            expansion_time_ms=hyde_result.generation_time_ms,
            expander_name=self.name,
            error=hyde_result.error,
        )

    async def expand_async(self, query: str, **kwargs) -> ExpansionResult:
        """Expand the query using HyDE asynchronously."""
        hyde_result = await self._get_expander().expand(query)

        return ExpansionResult(
            original_query=hyde_result.original_query,
            expanded_query=hyde_result.hypothetical_document or hyde_result.original_query,
            expanded=hyde_result.expanded,
            skip_reason=hyde_result.skip_reason or hyde_result.error,
            context_terms=[],
            metadata={
                "model_used": hyde_result.model_used,
                "hypothetical_document": hyde_result.hypothetical_document,
            },
            expansion_time_ms=hyde_result.generation_time_ms,
            expander_name=self.name,
            error=hyde_result.error,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            **super().get_config(),
            "enabled": settings.hyde_enabled,
            "model": getattr(settings, 'hyde_model', None),
        }

    def is_available(self) -> bool:
        return settings.hyde_enabled


__all__ = ["HyDEExpander"]
