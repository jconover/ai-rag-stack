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
from app.query_expansion import hyde_expander
from app.web_search import web_searcher

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
    # HyDE (Hypothetical Document Embeddings) fields
    hyde_used: bool = False
    hyde_time_ms: float = 0.0
    hyde_skipped_reason: Optional[str] = None
    # Web search fallback fields
    web_search_used: bool = False
    web_search_time_ms: float = 0.0
    web_search_results_count: int = 0
    web_search_trigger_reason: Optional[str] = None
    web_search_error: Optional[str] = None
    web_search_context: str = ""  # Formatted web results for context
    # Embedding cache tracking
    embedding_cache_hit: Optional[bool] = None


class RAGPipeline:
    # Default context token budget (leaves room for system prompt, query, and response)
    DEFAULT_MAX_CONTEXT_TOKENS = 4096

    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.default_model = settings.ollama_default_model

    def _count_tokens(self, text: str) -> int:
        """Estimate token count for a text string.

        Uses a rough approximation of ~4 characters per token, which is
        a reasonable estimate for English text with code snippets.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def _format_context(
        self,
        documents: List,
        web_context: str = "",
        max_context_tokens: Optional[int] = None
    ) -> str:
        """Format retrieved documents into context string with optional truncation.

        Args:
            documents: List of Document objects from vector store (ordered by relevance)
            web_context: Optional formatted web search results
            max_context_tokens: Maximum tokens for context. If None, uses DEFAULT_MAX_CONTEXT_TOKENS.
                               Documents are added in order of relevance until budget is exhausted.

        Returns:
            Combined context string for LLM, truncated to fit within token budget
        """
        if max_context_tokens is None:
            max_context_tokens = self.DEFAULT_MAX_CONTEXT_TOKENS

        context_parts = []
        current_tokens = 0
        separator = "\n---\n"
        separator_tokens = self._count_tokens(separator)

        # Reserve tokens for web context if present
        web_context_tokens = 0
        web_prefix = "\n\n--- Web Search Results ---\n\n"
        if web_context:
            web_context_tokens = self._count_tokens(web_prefix + web_context)

        available_tokens = max_context_tokens - web_context_tokens

        # Add local document context in order of relevance (higher-ranked first)
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            source_type = doc.metadata.get('source_type', 'Unknown')
            content = doc.page_content.strip()

            doc_text = f"[Source {i} - {source_type}]\n{content}\n"
            doc_tokens = self._count_tokens(doc_text)

            # Account for separator between documents
            needed_tokens = doc_tokens
            if context_parts:
                needed_tokens += separator_tokens

            # Check if adding this document would exceed the budget
            if current_tokens + needed_tokens > available_tokens:
                # Try to add a truncated version if we have room for at least some content
                remaining_tokens = available_tokens - current_tokens
                if context_parts:
                    remaining_tokens -= separator_tokens

                # Only truncate if we can include meaningful content (at least 50 tokens)
                if remaining_tokens >= 50:
                    # Estimate characters from tokens (reverse of _count_tokens)
                    remaining_chars = remaining_tokens * 4
                    header = f"[Source {i} - {source_type}]\n"
                    content_budget = remaining_chars - len(header) - 20  # -20 for safety margin

                    if content_budget > 100:
                        truncated_content = content[:content_budget] + "..."
                        doc_text = f"{header}{truncated_content}\n"
                        context_parts.append(doc_text)
                        logger.debug(
                            f"Context truncation: Document {i} truncated to {content_budget} chars "
                            f"(budget: {max_context_tokens} tokens)"
                        )
                break

            context_parts.append(doc_text)
            current_tokens += needed_tokens

        local_context = separator.join(context_parts) if context_parts else ""

        # Add web search context if available
        if web_context:
            if local_context:
                return f"{local_context}{web_prefix}{web_context}"
            else:
                return f"--- Web Search Results ---\n\n{web_context}"

        return local_context
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt defining the assistant's role and behavior."""
        return """You are an expert DevOps engineer assistant. Your role is to help users with DevOps, infrastructure, and programming questions.

## INSTRUCTIONS

- Answer based primarily on the provided context when available
- If the context doesn't fully answer the question, use your general knowledge but mention this
- Provide code examples when relevant, using proper markdown code blocks
- Be concise but thorough
- If you're unsure, say so

## OUTPUT FORMAT

- Use markdown formatting for readability (headers, lists, code blocks)
- Include source citations as [Source N] when referencing provided context
- Keep responses concise and focused - avoid unnecessary verbosity
- Structure longer responses with clear sections using markdown headers
- Use code blocks with language specifiers for all code examples (e.g., ```yaml, ```bash)

## SAFETY BOUNDARIES

- Only answer questions related to DevOps, infrastructure, cloud computing, CI/CD, containerization, orchestration, monitoring, and related technical topics
- For off-topic requests (personal advice, creative writing, general trivia, etc.), politely decline and redirect the user to ask DevOps-related questions
- Do not provide assistance with malicious activities such as unauthorized access, exploiting vulnerabilities, or bypassing security controls
- If a question is ambiguous, interpret it in the context of DevOps best practices"""

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

        # Phase 0: HyDE query expansion (if enabled)
        search_query = query
        if settings.hyde_enabled:
            hyde_result = hyde_expander.expand_sync(query)
            result.hyde_time_ms = hyde_result.generation_time_ms

            if hyde_result.expanded and hyde_result.hypothetical_document:
                # Use hypothetical document for retrieval (better semantic match)
                search_query = hyde_result.hypothetical_document
                result.hyde_used = True
                if settings.log_retrieval_details:
                    logger.info(
                        f"HyDE expanded query in {result.hyde_time_ms:.1f}ms: "
                        f"'{query[:50]}...' -> {len(hyde_result.hypothetical_document)} chars"
                    )
            else:
                result.hyde_skipped_reason = hyde_result.skip_reason or hyde_result.error

        # Phase 1: Vector search with scores (hybrid or dense-only)
        retrieval_start = time.perf_counter()
        try:
            # Use hybrid search if enabled, otherwise dense-only
            # Both methods now return (results, cache_hit) tuple
            if settings.hybrid_search_enabled:
                results_with_scores, cache_hit = vector_store.hybrid_search_with_cache_info(
                    query=search_query,
                    top_k=initial_top_k,
                    min_score=settings.min_similarity_score
                )
                result.hybrid_search_used = True
            else:
                results_with_scores, cache_hit = vector_store.search_with_cache_info(
                    query=search_query,
                    top_k=initial_top_k,
                    min_score=settings.min_similarity_score
                )
            result.embedding_cache_hit = cache_hit
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

        # Phase 3: Web search fallback (if enabled and local results are poor)
        if settings.web_search_enabled:
            # Use rerank scores if available (better quality signal than RRF similarity)
            if result.reranker_used and result.rerank_scores:
                avg_rerank = sum(result.rerank_scores) / len(result.rerank_scores)
                # Rerank scores: positive = relevant, negative = irrelevant
                # Trigger web search if avg rerank score is negative (poor relevance)
                if avg_rerank < 0:
                    should_search = True
                    reason = f"low_rerank_score_{avg_rerank:.2f}"
                else:
                    should_search = False
                    reason = None
            else:
                # Fallback to similarity scores if reranker not used
                avg_score = (
                    sum(result.similarity_scores) / len(result.similarity_scores)
                    if result.similarity_scores else 0.0
                )
                max_score = max(result.similarity_scores) if result.similarity_scores else 0.0

                should_search, reason = web_searcher.should_search(
                    avg_similarity_score=avg_score,
                    max_similarity_score=max_score,
                    result_count=result.final_count,
                )

            if should_search:
                result.web_search_trigger_reason = reason
                logger.info(f"Triggering web search fallback: {reason}")

                try:
                    # Use synchronous search (httpx.Client, safe in any context)
                    web_response = web_searcher.search_sync(query)

                    result.web_search_time_ms = web_response.search_time_ms
                    result.web_search_used = web_response.triggered and not web_response.error

                    if web_response.results:
                        result.web_search_results_count = len(web_response.results)
                        result.web_search_context = web_searcher.format_for_context(web_response.results)
                        logger.info(
                            f"Web search returned {len(web_response.results)} results "
                            f"in {web_response.search_time_ms:.0f}ms"
                        )
                    elif web_response.error:
                        result.web_search_error = web_response.error
                        logger.warning(f"Web search error: {web_response.error}")

                except Exception as e:
                    result.web_search_error = str(e)
                    logger.error(f"Web search fallback failed: {e}")

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

    async def _retrieve_with_scores_async(self, query: str, model: str = None) -> RetrievalResult:
        """Async wrapper for _retrieve_with_scores that runs in a thread pool executor.

        This prevents CPU-bound operations (vector search, reranking) from blocking
        the event loop in async contexts like FastAPI endpoints.

        Args:
            query: The search query string
            model: The LLM model being used (for metrics logging)

        Returns:
            RetrievalResult with documents, scores, timing, and metadata
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._retrieve_with_scores, query, model)

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
            'hyde_used': result.hyde_used,
            'hyde_time_ms': round(result.hyde_time_ms, 2) if result.hyde_used else None,
            'web_search_used': result.web_search_used,
            'web_search_reason': result.web_search_trigger_reason,
            'web_search_results': result.web_search_results_count if result.web_search_used else None,
            'web_search_time_ms': round(result.web_search_time_ms, 2) if result.web_search_used else None,
            'embedding_cache_hit': result.embedding_cache_hit,
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
        if result.web_search_error:
            metrics['web_search_error'] = result.web_search_error

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

        # Retrieve relevant context if RAG is enabled (async to avoid blocking event loop)
        if use_rag:
            try:
                retrieval_result = await self._retrieve_with_scores_async(query, model)
                context_str = self._format_context(
                    retrieval_result.documents,
                    web_context=retrieval_result.web_search_context
                )
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
                'context_used': bool(retrieval_result.documents) or bool(retrieval_result.web_search_context),
                'sources': (
                    self._format_sources_with_scores(retrieval_result)
                    if retrieval_result.documents else None
                ),
                'reranker_enabled': settings.reranker_enabled,
            }

            # Include retrieval metrics if enabled
            if settings.enable_retrieval_metrics and (retrieval_result.documents or retrieval_result.web_search_used):
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

        # Retrieve relevant context if RAG is enabled (async to avoid blocking event loop)
        if use_rag:
            try:
                retrieval_result = await self._retrieve_with_scores_async(query, model)
                context_str = self._format_context(
                    retrieval_result.documents,
                    web_context=retrieval_result.web_search_context
                )
            except Exception as e:
                logger.error("Error retrieving context: %s", e)

        # Build messages with proper system/user separation
        messages = self._build_messages(query, context_str)

        # Build metadata response with sources including scores
        metadata = {
            'type': 'metadata',
            'model': model,
            'context_used': bool(retrieval_result.documents) or bool(retrieval_result.web_search_context),
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
