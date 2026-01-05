"""RAG (Retrieval-Augmented Generation) pipeline with reranking, score-aware retrieval, and metrics.

This module implements a production-ready RAG pipeline with:
- Score-aware vector retrieval with configurable thresholds
- Optional cross-encoder reranking for improved relevance
- Comprehensive metrics logging for observability
- Streaming response support

Flow:
    Query -> Vector Search (top 20) -> Rerank (top 5) -> Build Context -> LLM
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import ollama

from app.config import settings
from app.vectorstore import vector_store
from app.reranker import rerank_documents, get_reranker
from app.metrics import retrieval_metrics_logger, RetrievalTimer, RetrievalMetrics

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval results with scores and performance metrics."""
    documents: List[Any] = field(default_factory=list)
    similarity_scores: List[float] = field(default_factory=list)
    rerank_scores: List[float] = field(default_factory=list)
    initial_count: int = 0
    final_count: int = 0
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0
    reranker_used: bool = False
    reranker_model: Optional[str] = None
    retrieval_error: Optional[str] = None
    rerank_error: Optional[str] = None
    # Hybrid search fields
    hybrid_search_used: bool = False
    dense_count: int = 0
    sparse_count: int = 0


class RAGPipeline:
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.default_model = settings.ollama_default_model
    
    def _format_context(self, documents: List) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            source_type = doc.metadata.get('source_type', 'Unknown')
            content = doc.page_content.strip()
            
            context_parts.append(
                f"[Source {i} - {source_type}]\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt defining the assistant's role and behavior."""
        return """You are an expert DevOps engineer assistant. Your role is to help users with DevOps, infrastructure, and programming questions.

Instructions:
- Answer based primarily on the provided context when available
- If the context doesn't fully answer the question, use your general knowledge but mention this
- Provide code examples when relevant, using proper markdown code blocks
- Be concise but thorough
- If you're unsure, say so
- When citing sources, reference them as [Source N]"""

    def _get_user_prompt(self, query: str, context: str) -> str:
        """Build the user prompt with context and query."""
        if context:
            return f"""Context from documentation:
{context}

Question: {query}"""
        else:
            return query

    def _build_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        """Build messages list with proper system/user role separation."""
        return [
            {'role': 'system', 'content': self._get_system_prompt()},
            {'role': 'user', 'content': self._get_user_prompt(query, context)}
        ]
    
    def _retrieve_with_scores(self, query: str, model: str = None) -> RetrievalResult:
        """Retrieve documents with similarity scores and optional reranking.

        Implements the core retrieval flow:
        1. Vector search for initial candidates (retrieval_top_k if reranking, else top_k)
        2. Apply minimum similarity score threshold
        3. Optional: Rerank candidates with cross-encoder
        4. Return final top_k results with all scores

        Args:
            query: The search query string
            model: The LLM model being used (for metrics logging)

        Returns:
            RetrievalResult with documents, scores, timing, and metadata
        """
        model = model or self.default_model
        result = RetrievalResult()
        total_start = time.perf_counter()

        # Determine initial retrieval count based on whether reranking is enabled
        if settings.reranker_enabled:
            initial_top_k = settings.retrieval_top_k
        else:
            initial_top_k = settings.top_k_results

        # Phase 1: Vector search with scores (hybrid or dense-only)
        retrieval_start = time.perf_counter()
        try:
            # Use hybrid search if enabled, otherwise dense-only
            if settings.hybrid_search_enabled:
                results_with_scores = vector_store.hybrid_search_with_scores(
                    query=query,
                    top_k=initial_top_k,
                    min_score=settings.min_similarity_score
                )
                result.hybrid_search_used = True
            else:
                results_with_scores = vector_store.search_with_scores(
                    query=query,
                    top_k=initial_top_k,
                    min_score=settings.min_similarity_score
                )
            result.retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
            result.initial_count = len(results_with_scores)

        except Exception as e:
            result.retrieval_error = str(e)
            logger.error(f"Vector search failed: {e}")
            result.total_time_ms = (time.perf_counter() - total_start) * 1000
            return result

        if not results_with_scores:
            result.total_time_ms = (time.perf_counter() - total_start) * 1000
            return result

        # Extract documents and similarity scores
        documents = [doc for doc, _ in results_with_scores]
        similarity_scores = [score for _, score in results_with_scores]

        if settings.log_retrieval_details:
            score_preview = ", ".join(f"{s:.3f}" for s in similarity_scores[:5])
            logger.info(
                f"Vector search: {len(documents)} results in {result.retrieval_time_ms:.1f}ms, "
                f"scores=[{score_preview}]"
            )

        # Phase 2: Reranking (if enabled)
        if settings.reranker_enabled:
            rerank_start = time.perf_counter()
            try:
                reranked_docs = rerank_documents(
                    query=query,
                    documents=documents,
                    top_k=settings.reranker_top_k,
                )
                result.rerank_time_ms = (time.perf_counter() - rerank_start) * 1000
                result.reranker_used = True
                result.reranker_model = settings.reranker_model

                # Extract rerank scores from document metadata
                rerank_scores = []
                for doc in reranked_docs:
                    score = doc.metadata.get('rerank_score', 0.0)
                    rerank_scores.append(score)

                # Map similarity scores to reranked order using content hash
                doc_to_sim_score = {}
                for doc, score in results_with_scores:
                    key = hash(doc.page_content[:100])
                    doc_to_sim_score[key] = score

                # Rebuild similarity scores in reranked order
                new_similarity_scores = []
                for doc in reranked_docs:
                    key = hash(doc.page_content[:100])
                    new_similarity_scores.append(doc_to_sim_score.get(key, 0.0))

                documents = reranked_docs
                similarity_scores = new_similarity_scores
                result.rerank_scores = rerank_scores

                if settings.log_retrieval_details:
                    rerank_preview = ", ".join(f"{s:.3f}" for s in rerank_scores[:5])
                    logger.info(
                        f"Reranking: {len(reranked_docs)} results in {result.rerank_time_ms:.1f}ms, "
                        f"scores=[{rerank_preview}]"
                    )

            except Exception as e:
                result.rerank_error = str(e)
                logger.error(f"Reranking failed: {e}, using original order")
                # Fall back to original results without reranking
                documents = documents[:settings.top_k_results]
                similarity_scores = similarity_scores[:settings.top_k_results]
        else:
            # No reranking, just take top_k results
            documents = documents[:settings.top_k_results]
            similarity_scores = similarity_scores[:settings.top_k_results]

        result.documents = documents
        result.similarity_scores = similarity_scores
        result.final_count = len(documents)
        result.total_time_ms = (time.perf_counter() - total_start) * 1000

        # Log metrics if enabled
        if settings.enable_retrieval_metrics:
            scores_to_log = result.rerank_scores if result.reranker_used else result.similarity_scores
            try:
                retrieval_metrics_logger.log_retrieval(
                    query=query,
                    model=model,
                    scores=scores_to_log,
                    top_k=settings.top_k_results,
                    latency_ms=result.total_time_ms,
                    score_threshold=settings.min_similarity_score,
                    filtered_count=0
                )
            except Exception as e:
                logger.warning(f"Failed to log retrieval metrics: {e}")

        return result

    def _retrieve_with_metrics(
        self,
        query: str,
        model: str = None
    ) -> Tuple[List, Optional[RetrievalMetrics]]:
        """Backwards-compatible wrapper around _retrieve_with_scores.

        Returns documents and a simplified metrics object for existing code.
        """
        result = self._retrieve_with_scores(query, model)

        # Build simplified metrics for backwards compatibility
        metrics = None
        if settings.enable_retrieval_metrics and result.documents:
            scores = result.rerank_scores if result.reranker_used else result.similarity_scores
            if scores:
                import statistics
                metrics = RetrievalMetrics(
                    timestamp="",
                    query_hash="",
                    query_preview=query[:100],
                    model=model or self.default_model,
                    top_k=settings.top_k_results,
                    num_results=len(result.documents),
                    scores=scores,
                    latency_ms=result.total_time_ms,
                    score_threshold=settings.min_similarity_score,
                    filtered_count=0,
                    score_min=min(scores) if scores else None,
                    score_max=max(scores) if scores else None,
                    score_mean=statistics.mean(scores) if scores else None,
                )

        return result.documents, metrics

    def _retrieve_and_rerank(self, query: str) -> List:
        """Retrieve documents and optionally rerank them (backwards compatible).

        This is a convenience wrapper around _retrieve_with_scores that
        discards the detailed result for backwards compatibility.

        Args:
            query: The search query string

        Returns:
            List of Document objects, reranked if enabled
        """
        result = self._retrieve_with_scores(query)
        return result.documents

    def _format_sources_with_scores(self, result: RetrievalResult) -> List[Dict[str, Any]]:
        """Format retrieval results into source metadata with scores for API response.

        Each source includes:
        - source: Document source path/name
        - source_type: Type of documentation (kubernetes, terraform, etc.)
        - content_preview: First 200 chars of content
        - rank: Position in final results (1-indexed)
        - similarity_score: Vector similarity score (0-1)
        - rerank_score: Cross-encoder relevance score (if reranking used)

        Args:
            result: RetrievalResult from _retrieve_with_scores

        Returns:
            List of source dictionaries with scores and metadata
        """
        sources = []

        for i, doc in enumerate(result.documents):
            source_info = {
                'source': doc.metadata.get('source', 'Unknown'),
                'source_type': doc.metadata.get('source_type', 'Unknown'),
                'content_preview': (
                    doc.page_content[:200] + '...'
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
                'rank': i + 1,
            }

            # Add similarity score
            if i < len(result.similarity_scores):
                source_info['similarity_score'] = round(result.similarity_scores[i], 4)

            # Add rerank score if available
            if result.reranker_used and i < len(result.rerank_scores):
                source_info['rerank_score'] = round(result.rerank_scores[i], 4)

            sources.append(source_info)

        return sources

    def _build_retrieval_metrics_dict(self, result: RetrievalResult) -> Dict[str, Any]:
        """Build retrieval metrics dictionary for API response.

        Args:
            result: RetrievalResult from _retrieve_with_scores

        Returns:
            Dictionary with retrieval performance metrics
        """
        metrics = {
            'initial_candidates': result.initial_count,
            'final_results': result.final_count,
            'reranker_used': result.reranker_used,
            'reranker_model': result.reranker_model,
            'retrieval_time_ms': round(result.retrieval_time_ms, 2),
            'rerank_time_ms': round(result.rerank_time_ms, 2) if result.reranker_used else None,
            'total_time_ms': round(result.total_time_ms, 2),
            'hybrid_search_used': result.hybrid_search_used,
        }

        # Calculate average scores
        if result.similarity_scores:
            metrics['avg_similarity_score'] = round(
                sum(result.similarity_scores) / len(result.similarity_scores), 4
            )

        if result.rerank_scores:
            metrics['avg_rerank_score'] = round(
                sum(result.rerank_scores) / len(result.rerank_scores), 4
            )

        # Include any errors
        if result.retrieval_error:
            metrics['retrieval_error'] = result.retrieval_error
        if result.rerank_error:
            metrics['rerank_error'] = result.rerank_error

        return metrics

    def get_reranker_status(self) -> Dict[str, Any]:
        """Get reranker component status for health checks.

        Returns:
            Dictionary with:
            - enabled: Whether reranker is enabled in config
            - loaded: Whether model is loaded in memory
            - model: Model name if loaded
            - device: Device (cpu/cuda) if loaded
            - error: Any error message
        """
        reranker = get_reranker()

        if reranker is None:
            return {
                'enabled': settings.reranker_enabled,
                'loaded': False,
                'model': None,
                'device': None,
                'error': None if not settings.reranker_enabled else 'Reranker not initialized'
            }

        return {
            'enabled': True,
            'loaded': True,
            **reranker.get_model_info()
        }

    async def generate_response(
        self,
        query: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_rag: bool = True,
    ) -> Dict[str, Any]:
        """Generate response using RAG pipeline with reranking and metrics.

        Flow:
            Query -> Vector Search (top 20) -> Rerank (top 5) -> Build Context -> LLM

        Args:
            query: User question/message
            model: Ollama model to use (default from settings)
            temperature: Generation temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            use_rag: Whether to use RAG context

        Returns:
            Dictionary with:
            - response: LLM generated answer
            - model: Model used
            - context_used: Whether RAG context was used
            - sources: List of source documents with scores
            - retrieval_metrics: Performance and quality metrics (if enabled)
        """
        model = model or self.default_model
        retrieval_result = RetrievalResult()
        context_str = ""

        # Retrieve relevant context if RAG is enabled
        if use_rag:
            try:
                retrieval_result = self._retrieve_with_scores(query, model)
                context_str = self._format_context(retrieval_result.documents)
            except Exception as e:
                logger.error("Error retrieving context: %s", e)

        # Build messages with proper system/user separation
        messages = self._build_messages(query, context_str)

        # Generate response using Ollama
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                }
            )

            answer = response['message']['content']

            # Build response with sources including scores
            result = {
                'response': answer,
                'model': model,
                'context_used': bool(retrieval_result.documents),
                'sources': (
                    self._format_sources_with_scores(retrieval_result)
                    if retrieval_result.documents else None
                ),
                'reranker_enabled': settings.reranker_enabled,
            }

            # Include retrieval metrics if enabled
            if settings.enable_retrieval_metrics and retrieval_result.documents:
                result['retrieval_metrics'] = self._build_retrieval_metrics_dict(retrieval_result)

            return result

        except Exception as e:
            raise Exception("Error generating response: {}".format(str(e)))
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models"""
        try:
            models = ollama.list()
            return models.get('models', [])
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def is_ollama_connected(self) -> bool:
        """Check if Ollama is accessible"""
        try:
            ollama.list()
            return True
        except:
            return False

    def _run_ollama_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Run synchronous Ollama streaming in a thread and put results in async queue.

        This method runs in a separate thread to avoid blocking the event loop.
        """
        try:
            stream = ollama.chat(
                model=model,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                },
                stream=True
            )

            for chunk in stream:
                if chunk.get('message', {}).get('content'):
                    # Thread-safe way to put item in async queue
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {'type': 'content', 'content': chunk['message']['content']}
                    )

            # Signal completion
            loop.call_soon_threadsafe(queue.put_nowait, {'type': 'done'})

        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {'type': 'error', 'error': str(e)}
            )

    async def generate_response_stream(
        self,
        query: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_rag: bool = True,
    ):
        """Generate streaming response using RAG pipeline with reranking.

        Flow:
            Query -> Vector Search (top 20) -> Rerank (top 5) -> Build Context -> LLM (streaming)

        This async generator properly yields control back to the event loop
        by running the synchronous Ollama streaming in a thread pool.

        Yields:
            Chunks with type field:
            - metadata: Initial response with sources, model, metrics
            - content: Token chunks from LLM
            - done: Completion signal
            - error: Error information if something fails
        """
        model = model or self.default_model
        retrieval_result = RetrievalResult()
        context_str = ""

        # Retrieve relevant context if RAG is enabled
        if use_rag:
            try:
                retrieval_result = self._retrieve_with_scores(query, model)
                context_str = self._format_context(retrieval_result.documents)
            except Exception as e:
                logger.error("Error retrieving context: %s", e)

        # Build messages with proper system/user separation
        messages = self._build_messages(query, context_str)

        # Build metadata response with sources including scores
        metadata = {
            'type': 'metadata',
            'model': model,
            'context_used': bool(retrieval_result.documents),
            'sources': (
                self._format_sources_with_scores(retrieval_result)
                if retrieval_result.documents else None
            ),
            'reranker_enabled': settings.reranker_enabled,
        }

        # Include retrieval metrics if enabled
        if settings.enable_retrieval_metrics and retrieval_result.documents:
            metadata['retrieval_metrics'] = self._build_retrieval_metrics_dict(retrieval_result)

        # Yield metadata first
        yield metadata

        # Generate streaming response using Ollama in a thread pool
        # This prevents blocking the event loop
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Run the synchronous Ollama streaming in a thread pool
        thread_task = loop.run_in_executor(
            None,  # Use default executor (ThreadPoolExecutor)
            self._run_ollama_stream,
            model,
            messages,
            temperature,
            max_tokens,
            queue,
            loop,
        )

        # Consume chunks from the queue as they arrive
        try:
            while True:
                # Wait for next chunk from the thread, yielding control to event loop
                chunk = await queue.get()

                if chunk['type'] == 'done':
                    yield chunk
                    break
                elif chunk['type'] == 'error':
                    yield chunk
                    break
                else:
                    yield chunk
        finally:
            # Ensure the thread task completes
            await thread_task


# Singleton instance
rag_pipeline = RAGPipeline()
