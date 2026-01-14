"""Composite query expander for chaining multiple expanders.

Allows multiple query expansion strategies to be applied in sequence.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple

from app.retrieval.expanders.base import QueryExpander, ExpansionResult

logger = logging.getLogger(__name__)


class CompositeExpander(QueryExpander):
    """Composite expander that chains multiple expanders in sequence.

    Applies expanders in order, passing the output of one as input to the next.
    """

    def __init__(self, expanders: List[QueryExpander]):
        if not expanders:
            raise ValueError("CompositeExpander requires at least one expander")
        self._expanders = expanders

    @property
    def name(self) -> str:
        expander_names = [exp.name for exp in self._expanders]
        return f"composite[{'+'.join(expander_names)}]"

    def should_expand(self, query: str, **kwargs) -> Tuple[bool, Optional[str]]:
        """Returns True if any expander should expand."""
        for expander in self._expanders:
            if expander.is_available():
                should, _ = expander.should_expand(query, **kwargs)
                if should:
                    return True, None
        return False, "no_expanders_should_run"

    def expand(self, query: str, **kwargs) -> ExpansionResult:
        """Expand the query by chaining expanders synchronously."""
        current_query = query
        all_context_terms = []
        all_metadata = {}
        total_time_ms = 0.0

        for expander in self._expanders:
            if not expander.is_available():
                logger.debug(f"Skipping unavailable expander: {expander.name}")
                continue

            result = expander.expand(current_query, **kwargs)

            total_time_ms += result.expansion_time_ms
            all_context_terms.extend(result.context_terms)
            all_metadata[expander.name] = result.to_dict()

            if result.expanded:
                current_query = result.expanded_query
                logger.debug(f"{expander.name} expanded query")

        return ExpansionResult(
            original_query=query,
            expanded_query=current_query,
            expanded=(current_query != query),
            context_terms=all_context_terms,
            metadata={"expanders": all_metadata},
            expansion_time_ms=total_time_ms,
            expander_name=self.name,
        )

    async def expand_async(self, query: str, **kwargs) -> ExpansionResult:
        """Expand the query by chaining expanders asynchronously."""
        current_query = query
        all_context_terms = []
        all_metadata = {}
        total_time_ms = 0.0

        for expander in self._expanders:
            if not expander.is_available():
                logger.debug(f"Skipping unavailable expander: {expander.name}")
                continue

            result = await expander.expand_async(current_query, **kwargs)

            total_time_ms += result.expansion_time_ms
            all_context_terms.extend(result.context_terms)
            all_metadata[expander.name] = result.to_dict()

            if result.expanded:
                current_query = result.expanded_query
                logger.debug(f"{expander.name} expanded query")

        return ExpansionResult(
            original_query=query,
            expanded_query=current_query,
            expanded=(current_query != query),
            context_terms=all_context_terms,
            metadata={"expanders": all_metadata},
            expansion_time_ms=total_time_ms,
            expander_name=self.name,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expanders": [exp.get_config() for exp in self._expanders],
        }

    def is_available(self) -> bool:
        return any(exp.is_available() for exp in self._expanders)


__all__ = ["CompositeExpander"]
