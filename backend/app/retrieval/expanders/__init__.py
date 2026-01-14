"""Query expanders for the retrieval pipeline.

This package provides modular query expansion strategies that can be
composed together to improve retrieval quality.

Available expanders:
- BaseExpander: Abstract base class with common functionality
- HyDEExpander: Hypothetical Document Embeddings for vague queries
- ConversationExpander: Resolves pronouns using conversation history
- CompositeExpander: Chains multiple expanders in sequence

Usage:
    from app.retrieval.expanders import HyDEExpander, ConversationExpander, CompositeExpander

    # Single expander
    hyde = HyDEExpander()
    result = await hyde.expand("kubernetes networking")

    # Composite pipeline
    pipeline = CompositeExpander([
        ConversationExpander(),
        HyDEExpander(),
    ])
    result = await pipeline.expand("how do I scale it?", conversation_history=history)
"""

from app.retrieval.expanders.base import QueryExpander, BaseExpander, ExpansionResult
from app.retrieval.expanders.hyde import HyDEExpander
from app.retrieval.expanders.conversation import ConversationExpander
from app.retrieval.expanders.composite import CompositeExpander

__all__ = [
    "QueryExpander",
    "BaseExpander",
    "ExpansionResult",
    "HyDEExpander",
    "ConversationExpander",
    "CompositeExpander",
]
