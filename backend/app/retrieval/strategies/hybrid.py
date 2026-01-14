"""Hybrid retrieval strategy combining dense vector search with BM25 sparse retrieval.

This strategy implements the hybrid search approach used in the RAG pipeline,
with support for reranking and web search fallback.
"""

import logging
import time
from typing import Optional, Dict, Any

from app.retrieval.base import RetrievalStrategy, RetrievalResult
from app.config import settings

logger = logging.getLogger(__name__)


class HybridRetrievalStrategy(RetrievalStrategy):
    """Hybrid retrieval combining dense vector search with BM25 sparse retrieval.

    This strategy implements the complete retrieval flow:
    1. Hybrid search (vector + BM25) or dense-only vector search
    2. Optional cross-encoder reranking
    3. Score filtering
    4. Web search fallback for low-quality results
    """

    def __init__(self):
        self._vector_store = None
        self._reranker = None
        self._web_searcher = None

    def _get_vector_store(self):
        """Lazy load vector store to avoid circular imports."""
        if self._vector_store is None:
            from app.vectorstore import vector_store
            self._vector_store = vector_store
        return self._vector_store

    def _get_reranker(self):
        """Lazy load reranker."""
        if self._reranker is None and settings.reranker_enabled:
            from app.reranker import rerank_documents
            self._reranker = rerank_documents
        return self._reranker

    def _get_web_searcher(self):
        """Lazy load web searcher."""
        if self._web_searcher is None and settings.web_search_enabled:
            from app.web_search import web_searcher
            self._web_searcher = web_searcher
        return self._web_searcher

    @property
    def name(self) -> str:
        if settings.hybrid_search_enabled:
            if settings.reranker_enabled:
                return "hybrid_with_reranking"
            return "hybrid"
        elif settings.reranker_enabled:
            return "vector_with_reranking"
        return "vector"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve documents synchronously."""
        result = RetrievalResult(strategy_name=self.name)
        start_time = time.perf_counter()

        if min_score is None:
            min_score = settings.min_similarity_score

        initial_top_k = settings.retrieval_top_k if settings.reranker_enabled else top_k
        vector_store = self._get_vector_store()

        try:
            if settings.hybrid_search_enabled:
                results_with_scores, cache_hit = vector_store.hybrid_search_with_cache_info(
                    query=query,
                    top_k=initial_top_k,
                    min_score=min_score
                )
            else:
                results_with_scores, cache_hit = vector_store.search_with_cache_info(
                    query=query,
                    top_k=initial_top_k,
                    min_score=min_score
                )

            result.cache_hit = cache_hit
            result.metadata['initial_count'] = len(results_with_scores)

        except Exception as e:
            result.error = str(e)
            logger.error(f"Vector search failed: {e}")
            result.timing_ms = (time.perf_counter() - start_time) * 1000
            return result

        if not results_with_scores:
            result.timing_ms = (time.perf_counter() - start_time) * 1000
            return result

        documents = [doc for doc, _ in results_with_scores]
        scores = [score for _, score in results_with_scores]

        # Reranking
        if settings.reranker_enabled:
            reranker = self._get_reranker()
            if reranker:
                try:
                    reranked_docs = reranker(query, documents, settings.reranker_top_k)
                    result.metadata['reranker_used'] = True

                    # Extract rerank scores
                    rerank_scores = []
                    for doc in reranked_docs:
                        rerank_score = doc.metadata.get('rerank_score', 0.0)
                        rerank_scores.append(rerank_score)

                    # Filter by min_rerank_score
                    filtered_docs = []
                    filtered_scores = []
                    for doc, score in zip(reranked_docs, rerank_scores):
                        if score >= settings.min_rerank_score:
                            filtered_docs.append(doc)
                            filtered_scores.append(score)

                    if not filtered_docs and reranked_docs:
                        filtered_docs = [reranked_docs[0]]
                        filtered_scores = [rerank_scores[0]]

                    documents = filtered_docs
                    scores = filtered_scores

                except Exception as e:
                    logger.error(f"Reranking failed: {e}")
                    result.metadata['rerank_error'] = str(e)
                    documents = documents[:top_k]
                    scores = scores[:top_k]
        else:
            documents = documents[:top_k]
            scores = scores[:top_k]

        result.documents = documents
        result.scores = scores

        # Web search fallback
        if settings.web_search_enabled and documents:
            avg_score = sum(scores) / len(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0
            web_searcher = self._get_web_searcher()

            if web_searcher:
                should_search, reason = web_searcher.should_search(
                    avg_similarity_score=avg_score,
                    max_similarity_score=max_score,
                    result_count=len(documents)
                )

                if should_search:
                    try:
                        web_response = web_searcher.search_sync(query)
                        result.metadata['web_search_used'] = web_response.triggered and not web_response.error
                        if web_response.results:
                            result.metadata['web_search_context'] = web_searcher.format_for_context(
                                web_response.results
                            )
                    except Exception as e:
                        logger.error(f"Web search failed: {e}")
                        result.metadata['web_search_error'] = str(e)

        result.timing_ms = (time.perf_counter() - start_time) * 1000
        return result

    async def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve documents asynchronously."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.retrieve(query, top_k, min_score, **kwargs)
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hybrid_search_enabled": settings.hybrid_search_enabled,
            "reranker_enabled": settings.reranker_enabled,
            "retrieval_top_k": settings.retrieval_top_k,
            "min_similarity_score": settings.min_similarity_score,
            "web_search_enabled": settings.web_search_enabled,
        }

    def is_available(self) -> bool:
        try:
            return self._get_vector_store() is not None
        except Exception:
            return False


__all__ = ["HybridRetrievalStrategy"]
