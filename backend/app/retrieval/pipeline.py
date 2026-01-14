"""RAG Pipeline Orchestrator.

Coordinates retrieval strategies, query expanders, and response generators
to provide a unified interface for the RAG pipeline.
"""

import logging
import time
from typing import Optional, List, Dict, Any, AsyncIterator

from app.retrieval.base import RetrievalResult
from app.retrieval.expanders.base import QueryExpander, ExpansionResult
from app.retrieval.strategies.hybrid import HybridRetrievalStrategy
from app.retrieval.generators.base import ResponseGenerator, GenerationResult
from app.retrieval.generators.ollama import OllamaGenerator
from app.config import settings

logger = logging.getLogger(__name__)


class RAGPipelineOrchestrator:
    """Orchestrates the complete RAG pipeline.

    Coordinates:
    1. Query expansion (conversation context, HyDE)
    2. Document retrieval (hybrid search, reranking)
    3. Response generation (Ollama LLM)

    Example:
        orchestrator = RAGPipelineOrchestrator()

        result = await orchestrator.process_query(
            query="How do I create a Kubernetes deployment?",
            conversation_history=history,
            model="llama3.1:8b"
        )

        print(result["response"])
        print(result["sources"])
    """

    def __init__(
        self,
        retrieval_strategy: Optional[HybridRetrievalStrategy] = None,
        generator: Optional[ResponseGenerator] = None,
        expanders: Optional[List[QueryExpander]] = None,
    ):
        """Initialize the orchestrator.

        Args:
            retrieval_strategy: Strategy for document retrieval
            generator: Generator for response creation
            expanders: List of query expanders to apply
        """
        self.retrieval_strategy = retrieval_strategy or HybridRetrievalStrategy()
        self.generator = generator or OllamaGenerator()
        self.expanders = expanders or []

    async def _expand_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> ExpansionResult:
        """Apply query expanders in sequence.

        Args:
            query: Original query
            conversation_history: Prior conversation messages

        Returns:
            ExpansionResult with potentially expanded query
        """
        current_query = query
        all_metadata = {}

        for expander in self.expanders:
            if not expander.is_available():
                continue

            try:
                result = await expander.expand_async(
                    current_query,
                    conversation_history=conversation_history
                )

                all_metadata[expander.name] = result.to_dict()

                if result.expanded:
                    current_query = result.expanded_query
                    logger.debug(f"Query expanded by {expander.name}")

            except Exception as e:
                logger.warning(f"Expander {expander.name} failed: {e}")
                all_metadata[expander.name] = {"error": str(e)}

        return ExpansionResult(
            original_query=query,
            expanded_query=current_query,
            expanded=(current_query != query),
            metadata={"expanders": all_metadata},
            expander_name="pipeline",
        )

    def _format_context(self, documents: List[Any]) -> str:
        """Format retrieved documents as context string.

        Args:
            documents: List of Document objects

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documentation found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()
            context_parts.append(f"[Source {i}: {source}]\n{content}")

        return "\n\n---\n\n".join(context_parts)

    def _format_sources(self, documents: List[Any], scores: List[float]) -> List[Dict[str, Any]]:
        """Format sources for API response.

        Args:
            documents: List of Document objects
            scores: Relevance scores

        Returns:
            List of source dictionaries
        """
        sources = []
        for doc, score in zip(documents, scores):
            sources.append({
                "content": doc.page_content[:500],
                "source": doc.metadata.get('source', 'Unknown'),
                "score": score,
                "metadata": doc.metadata,
            })
        return sources

    async def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        top_k: int = 5,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline.

        Args:
            query: User's question
            conversation_history: Prior conversation messages
            model: LLM model to use
            top_k: Number of documents to retrieve
            temperature: Generation temperature
            **kwargs: Additional parameters

        Returns:
            Dictionary with response, sources, and metadata
        """
        start_time = time.perf_counter()
        result = {
            "query": query,
            "response": "",
            "sources": [],
            "metadata": {},
        }

        # Phase 1: Query Expansion
        expansion_result = await self._expand_query(query, conversation_history)
        search_query = expansion_result.expanded_query
        result["metadata"]["expansion"] = expansion_result.to_dict()

        # Phase 2: Document Retrieval
        retrieval_result = await self.retrieval_strategy.retrieve_async(
            search_query,
            top_k=top_k,
            **kwargs
        )

        result["sources"] = self._format_sources(
            retrieval_result.documents,
            retrieval_result.scores
        )
        result["metadata"]["retrieval"] = retrieval_result.to_dict()

        if retrieval_result.is_empty:
            result["response"] = "I couldn't find relevant documentation to answer your question. Could you rephrase or provide more context?"
            result["metadata"]["total_time_ms"] = (time.perf_counter() - start_time) * 1000
            return result

        # Phase 3: Response Generation
        context = self._format_context(retrieval_result.documents)

        # Add web search context if available
        web_context = retrieval_result.metadata.get('web_search_context')
        if web_context:
            context = f"{context}\n\n---\n\n[Web Search Results]\n{web_context}"

        generation_result = await self.generator.generate(
            query=query,
            context=context,
            model=model,
            temperature=temperature,
            conversation_history=conversation_history,
        )

        result["response"] = generation_result.response
        result["model"] = generation_result.model
        result["metadata"]["generation"] = generation_result.to_dict()
        result["metadata"]["total_time_ms"] = (time.perf_counter() - start_time) * 1000

        return result

    async def process_query_stream(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        top_k: int = 5,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process a query with streaming response.

        Yields:
            Chunks with type field (metadata, sources, content, done, error)
        """
        start_time = time.perf_counter()

        # Phase 1: Query Expansion
        expansion_result = await self._expand_query(query, conversation_history)
        search_query = expansion_result.expanded_query

        # Phase 2: Document Retrieval
        retrieval_result = await self.retrieval_strategy.retrieve_async(
            search_query,
            top_k=top_k,
            **kwargs
        )

        # Yield metadata
        yield {
            "type": "metadata",
            "expansion": expansion_result.to_dict(),
            "retrieval": retrieval_result.to_dict(),
        }

        # Yield sources
        sources = self._format_sources(
            retrieval_result.documents,
            retrieval_result.scores
        )
        yield {
            "type": "sources",
            "sources": sources,
        }

        if retrieval_result.is_empty:
            yield {
                "type": "content",
                "content": "I couldn't find relevant documentation to answer your question.",
            }
            yield {
                "type": "done",
                "total_time_ms": (time.perf_counter() - start_time) * 1000,
            }
            return

        # Phase 3: Streaming Response Generation
        context = self._format_context(retrieval_result.documents)

        web_context = retrieval_result.metadata.get('web_search_context')
        if web_context:
            context = f"{context}\n\n---\n\n[Web Search Results]\n{web_context}"

        async for chunk in self.generator.generate_stream(
            query=query,
            context=context,
            model=model,
            temperature=temperature,
            conversation_history=conversation_history,
        ):
            yield chunk

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the orchestrator."""
        return {
            "retrieval_strategy": self.retrieval_strategy.get_config(),
            "generator": self.generator.get_config(),
            "expanders": [exp.get_config() for exp in self.expanders],
        }


# Create default orchestrator instance
def create_default_orchestrator() -> RAGPipelineOrchestrator:
    """Create orchestrator with default configuration."""
    expanders = []

    # Add conversation expander if enabled
    if getattr(settings, 'conversation_context_enabled', True):
        from app.retrieval.expanders.conversation import ConversationExpander
        expanders.append(ConversationExpander())

    # Add HyDE expander if enabled
    if settings.hyde_enabled:
        from app.retrieval.expanders.hyde import HyDEExpander
        expanders.append(HyDEExpander())

    return RAGPipelineOrchestrator(expanders=expanders)


__all__ = ["RAGPipelineOrchestrator", "create_default_orchestrator"]
