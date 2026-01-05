"""Qdrant vector store management with optimized search and performance tuning.

Supports both dense-only and hybrid (dense + sparse) search modes:
- Dense: Semantic similarity using sentence-transformers embeddings
- Sparse: BM25 keyword matching using fastembed
- Hybrid: Combines both using Reciprocal Rank Fusion (RRF)
"""
from typing import List, Optional, Tuple, Dict, Any, Union
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SearchParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    QuantizationSearchParams,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    SearchRequest,
    SparseVectorParams,
    SparseIndexParams,
    NamedVector,
    NamedSparseVector,
    SparseVector as QdrantSparseVector,
)
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from app.config import settings

logger = logging.getLogger(__name__)

# Sparse encoder for hybrid search (lazy loaded)
_sparse_encoder = None

def _get_sparse_encoder():
    """Get or initialize the sparse encoder for hybrid search."""
    global _sparse_encoder
    if _sparse_encoder is None and settings.hybrid_search_enabled:
        try:
            from app.sparse_encoder import SparseEncoder
            _sparse_encoder = SparseEncoder()
        except Exception as e:
            logger.warning(f"Failed to initialize sparse encoder: {e}")
    return _sparse_encoder


class VectorStore:
    """
    Optimized Qdrant vector store with:
    - Direct Qdrant client usage for fine-grained control
    - Similarity score retrieval with normalization
    - Filtering by score threshold and source type
    - HNSW parameter tuning for performance
    - Scalar quantization for memory efficiency
    - Batch search support for multiple queries
    - Payload indexing for efficient filtering
    """

    # HNSW Parameters for optimal performance
    # m: Number of edges per node (higher = better recall, more memory)
    # ef_construct: Search quality during index building (higher = better quality)
    HNSW_M = 16
    HNSW_EF_CONSTRUCT = 100
    # ef: Search quality at query time (higher = better recall, slower)
    HNSW_EF = 128

    # Quantization settings
    # Scalar quantization reduces vector storage by 4x with minimal quality loss
    ENABLE_QUANTIZATION = True
    QUANTIZATION_RESCORE = True
    QUANTIZATION_OVERSAMPLING = 2.0

    # Named vector constants for hybrid search
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,  # 30 second timeout for operations
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': settings.embedding_device}
        )
        # Warmup the embedding model to eliminate first-query cold-start latency
        self.embeddings.embed_query("warmup")
        self.collection_name = settings.qdrant_collection_name
        self._ensure_collection()
        self._ensure_payload_indexes()

    def _ensure_collection(self):
        """Create collection with optimized settings if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            # Create collection with optimized HNSW and quantization settings
            vectors_config = VectorParams(
                size=384,  # all-MiniLM-L6-v2 embedding dimension
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=self.HNSW_M,
                    ef_construct=self.HNSW_EF_CONSTRUCT,
                ),
            )

            # Configure scalar quantization for memory efficiency
            quantization_config = None
            if self.ENABLE_QUANTIZATION:
                quantization_config = ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,  # Exclude outliers
                        always_ram=True,  # Keep quantized vectors in RAM
                    )
                )

            # Optimized settings for search performance
            optimizers_config = OptimizersConfigDiff(
                indexing_threshold=20000,  # Start indexing after 20k vectors
                memmap_threshold=50000,  # Use mmap after 50k vectors
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                quantization_config=quantization_config,
                optimizers_config=optimizers_config,
            )
            logger.info(f"Created collection '{self.collection_name}' with optimized settings")
        else:
            # Update existing collection HNSW params if needed
            self._update_collection_params()

    def _update_collection_params(self):
        """Update existing collection with optimized HNSW parameters"""
        try:
            self.client.update_collection(
                collection_name=self.collection_name,
                hnsw_config=HnswConfigDiff(
                    m=self.HNSW_M,
                    ef_construct=self.HNSW_EF_CONSTRUCT,
                ),
            )
            logger.debug(f"Updated HNSW params for collection '{self.collection_name}'")
        except Exception as e:
            # Some params may not be updatable, which is fine
            logger.debug(f"Could not update collection params: {e}")

    def _ensure_payload_indexes(self):
        """Create payload indexes for efficient filtering"""
        try:
            # Index source_type for fast filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Created payload index on 'source_type'")
        except Exception as e:
            # Index may already exist
            logger.debug(f"Payload index 'source_type' creation skipped: {e}")

        try:
            # Index source for filtering by specific document
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Created payload index on 'source'")
        except Exception as e:
            logger.debug(f"Payload index 'source' creation skipped: {e}")

    def _normalize_score(self, score: float, distance: Distance = Distance.COSINE) -> float:
        """
        Normalize similarity score to 0-1 range.

        For COSINE distance in Qdrant:
        - Score is already similarity (1 - cosine_distance)
        - Range is typically [-1, 1], but with normalized vectors it's [0, 1]

        For DOT product:
        - Score can be any real number, needs different normalization

        For EUCLID:
        - Score is negative distance, need to convert
        """
        if distance == Distance.COSINE:
            # Qdrant returns cosine similarity directly for COSINE distance
            # Clamp to [0, 1] range for safety
            return max(0.0, min(1.0, score))
        elif distance == Distance.DOT:
            # Dot product can exceed 1 for non-normalized vectors
            # Use sigmoid-like normalization
            return max(0.0, min(1.0, (score + 1) / 2))
        elif distance == Distance.EUCLID:
            # Euclidean distance: score = -distance
            # Convert to similarity using exponential decay
            import math
            return math.exp(score)  # score is negative distance
        return score

    def _build_filter(
        self,
        source_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Optional[Filter]:
        """Build Qdrant filter from parameters"""
        filter_conditions = []

        if source_type:
            filter_conditions.append(
                FieldCondition(
                    key="source_type",
                    match=MatchValue(value=source_type),
                )
            )

        if source:
            filter_conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source),
                )
            )

        return Filter(must=filter_conditions) if filter_conditions else None

    def _get_search_params(self) -> SearchParams:
        """Get optimized search parameters"""
        search_params = SearchParams(
            hnsw_ef=self.HNSW_EF,  # Query-time ef parameter
            exact=False,  # Use approximate search for speed
        )

        # Add quantization params if enabled
        if self.ENABLE_QUANTIZATION:
            search_params.quantization = QuantizationSearchParams(
                rescore=self.QUANTIZATION_RESCORE,
                oversampling=self.QUANTIZATION_OVERSAMPLING,
            )

        return search_params

    def _process_search_results(
        self,
        results: List,
        min_score: float,
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        """Process raw Qdrant search results into Documents with scores"""
        documents_with_scores = []

        for hit in results:
            # Normalize score to 0-1 range
            normalized_score = self._normalize_score(hit.score)

            # Apply minimum score threshold
            if normalized_score < min_score:
                continue

            # Reconstruct Document from payload
            payload = hit.payload or {}
            doc = Document(
                page_content=payload.get('page_content', ''),
                metadata={
                    'source': payload.get('source', 'Unknown'),
                    'source_type': payload.get('source_type', 'Unknown'),
                    **{k: v for k, v in payload.items()
                       if k not in ('page_content', 'source', 'source_type')}
                }
            )
            documents_with_scores.append((doc, normalized_score))

            # Stop if we have enough results
            if len(documents_with_scores) >= top_k:
                break

        return documents_with_scores

    def search(
        self,
        query: str,
        top_k: int = None,
        min_score: float = None,
        source_type: str = None,
    ) -> List[Document]:
        """
        Search for relevant documents (without scores, for backwards compatibility).

        Args:
            query: Search query string
            top_k: Maximum number of results (default from settings)
            min_score: Minimum similarity score threshold (0-1)
            source_type: Filter by document source type (e.g., 'kubernetes', 'terraform')

        Returns:
            List of Documents
        """
        results = self.search_with_scores(
            query=query,
            top_k=top_k,
            min_score=min_score,
            source_type=source_type,
        )
        return [doc for doc, _ in results]

    def search_with_scores(
        self,
        query: str,
        top_k: int = None,
        min_score: float = None,
        source_type: str = None,
        source: str = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents and return with similarity scores.

        Uses direct Qdrant client for optimal performance with:
        - HNSW parameter tuning
        - Quantization for memory efficiency
        - Payload filtering with indexes

        Args:
            query: Search query string
            top_k: Maximum number of results
            min_score: Minimum similarity score threshold (0-1)
            source_type: Filter by document source type (e.g., 'kubernetes', 'terraform')
            source: Filter by specific source document path

        Returns:
            List of (Document, score) tuples, sorted by score descending
        """
        if top_k is None:
            top_k = settings.top_k_results
        if min_score is None:
            min_score = settings.min_similarity_score

        # Generate query embedding
        query_vector = self.embeddings.embed_query(query)

        # Build filter conditions
        query_filter = self._build_filter(source_type=source_type, source=source)

        # Get optimized search parameters
        search_params = self._get_search_params()

        # Fetch more results than needed to apply score filtering
        fetch_limit = min(top_k * 3, 100) if min_score > 0 else top_k

        # Execute search using Qdrant client directly
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=fetch_limit,
            query_filter=query_filter,
            search_params=search_params,
            with_payload=True,
        )

        return self._process_search_results(results, min_score, top_k)

    def search_batch(
        self,
        queries: List[str],
        top_k: int = None,
        min_score: float = None,
        source_type: str = None,
        source: str = None,
    ) -> List[List[Tuple[Document, float]]]:
        """
        Batch search for multiple queries efficiently.

        Uses Qdrant's batch search API for optimal performance when
        searching for multiple queries simultaneously. This is significantly
        faster than making individual search calls.

        Args:
            queries: List of search query strings
            top_k: Maximum number of results per query
            min_score: Minimum similarity score threshold (0-1)
            source_type: Filter by document source type
            source: Filter by specific source document path

        Returns:
            List of search results, one per query.
            Each result is List[Tuple[Document, float]]
        """
        if not queries:
            return []

        if top_k is None:
            top_k = settings.top_k_results
        if min_score is None:
            min_score = settings.min_similarity_score

        # Generate embeddings for all queries in batch
        # This is more efficient than embedding one at a time
        query_vectors = self.embeddings.embed_documents(queries)

        # Build filter conditions
        query_filter = self._build_filter(source_type=source_type, source=source)

        # Get optimized search parameters
        search_params = self._get_search_params()

        # Fetch more for score filtering
        fetch_limit = min(top_k * 3, 100) if min_score > 0 else top_k

        # Build batch search requests
        search_requests = [
            SearchRequest(
                vector=vec,
                limit=fetch_limit,
                filter=query_filter,
                params=search_params,
                with_payload=True,
            )
            for vec in query_vectors
        ]

        # Execute batch search
        batch_results = self.client.search_batch(
            collection_name=self.collection_name,
            requests=search_requests,
        )

        # Process all results
        all_results = []
        for results in batch_results:
            processed = self._process_search_results(results, min_score, top_k)
            all_results.append(processed)

        return all_results

    def hybrid_search_with_scores(
        self,
        query: str,
        top_k: int = None,
        min_score: float = None,
        source_type: str = None,
        source: str = None,
        alpha: float = None,
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid search combining dense (semantic) and sparse (BM25) retrieval.

        Uses Reciprocal Rank Fusion (RRF) to combine results from both
        search methods, providing benefits of:
        - Semantic understanding from dense vectors
        - Exact keyword matching from sparse vectors

        Falls back to dense-only search if:
        - Hybrid search is disabled in settings
        - Sparse encoder fails to initialize
        - Collection doesn't have sparse vectors

        Args:
            query: Search query string
            top_k: Maximum number of results
            min_score: Minimum similarity score threshold (0-1)
            source_type: Filter by document source type
            source: Filter by specific source document path
            alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
                   Default from settings (0.5 weights both equally)

        Returns:
            List of (Document, score) tuples, sorted by fused score descending
        """
        if top_k is None:
            top_k = settings.top_k_results
        if min_score is None:
            min_score = settings.min_similarity_score
        if alpha is None:
            alpha = settings.hybrid_search_alpha

        # Check if hybrid search should be used
        sparse_encoder = _get_sparse_encoder()
        if not settings.hybrid_search_enabled or sparse_encoder is None:
            logger.debug("Hybrid search disabled, falling back to dense-only")
            return self.search_with_scores(
                query=query,
                top_k=top_k,
                min_score=min_score,
                source_type=source_type,
                source=source,
            )

        # Generate dense embedding
        query_vector = self.embeddings.embed_query(query)

        # Generate sparse embedding
        try:
            sparse_vector = sparse_encoder.encode_query(query)
        except Exception as e:
            logger.warning(f"Sparse encoding failed: {e}, falling back to dense-only")
            return self.search_with_scores(
                query=query,
                top_k=top_k,
                min_score=min_score,
                source_type=source_type,
                source=source,
            )

        # Build filter conditions
        query_filter = self._build_filter(source_type=source_type, source=source)

        # Get optimized search parameters
        search_params = self._get_search_params()

        # Fetch more results than needed for RRF fusion
        fetch_limit = min(top_k * 4, 100)

        # Execute dense search
        try:
            dense_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=fetch_limit,
                query_filter=query_filter,
                search_params=search_params,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            dense_results = []

        # Execute sparse search
        sparse_results = []
        try:
            # Check if collection supports sparse vectors
            collection_info = self.client.get_collection(self.collection_name)
            has_sparse = False

            # Check for sparse vectors in collection config
            if hasattr(collection_info.config, 'params'):
                params = collection_info.config.params
                if hasattr(params, 'sparse_vectors') and params.sparse_vectors:
                    has_sparse = self.SPARSE_VECTOR_NAME in params.sparse_vectors

            if has_sparse and sparse_vector.indices:
                sparse_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=NamedSparseVector(
                        name=self.SPARSE_VECTOR_NAME,
                        vector=QdrantSparseVector(
                            indices=sparse_vector.indices,
                            values=sparse_vector.values,
                        )
                    ),
                    limit=fetch_limit,
                    query_filter=query_filter,
                    with_payload=True,
                )
            else:
                logger.debug("Collection doesn't support sparse vectors, using dense only")
        except Exception as e:
            logger.warning(f"Sparse search failed: {e}, using dense results only")

        # If no sparse results, return dense results only
        if not sparse_results:
            return self._process_search_results(dense_results, min_score, top_k)

        # Convert results to (Document, score) format for RRF
        dense_docs_scores = self._process_search_results(dense_results, 0.0, fetch_limit)
        sparse_docs_scores = self._process_search_results(sparse_results, 0.0, fetch_limit)

        # Apply Reciprocal Rank Fusion
        from app.sparse_encoder import reciprocal_rank_fusion

        fused_results = reciprocal_rank_fusion(
            dense_results=dense_docs_scores,
            sparse_results=sparse_docs_scores,
            k=settings.hybrid_rrf_k,
            alpha=alpha,
        )

        # Apply minimum score threshold and limit
        filtered_results = [
            (doc, score) for doc, score in fused_results
            if score >= min_score * 0.01  # RRF scores are much smaller
        ][:top_k]

        logger.debug(
            f"Hybrid search: {len(dense_docs_scores)} dense, {len(sparse_docs_scores)} sparse, "
            f"{len(filtered_results)} fused results"
        )

        return filtered_results

    def get_stats(self) -> dict:
        """Get collection statistics with detailed information"""
        try:
            collection_info = self.client.get_collection(self.collection_name)

            # Get optimizer status
            optimizer_status = "unknown"
            if collection_info.optimizer_status:
                optimizer_status = str(collection_info.optimizer_status.status)

            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count or 0,
                "points_count": collection_info.points_count or 0,
                "indexed_vectors_count": getattr(collection_info, 'indexed_vectors_count', None) or 0,
                "optimizer_status": optimizer_status,
                "status": str(collection_info.status),
                "config": {
                    "hnsw_m": self.HNSW_M,
                    "hnsw_ef_construct": self.HNSW_EF_CONSTRUCT,
                    "hnsw_ef_search": self.HNSW_EF,
                    "quantization_enabled": self.ENABLE_QUANTIZATION,
                }
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "vectors_count": 0,
                "points_count": 0,
                "error": str(e)
            }

    def get_source_types(self) -> List[str]:
        """Get list of unique source types in the collection"""
        try:
            # Use scroll to get unique source types
            # This is more efficient than fetching all points
            result, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=["source_type"],
                with_vectors=False,
            )

            source_types = set()
            for point in result:
                if point.payload and 'source_type' in point.payload:
                    source_types.add(point.payload['source_type'])

            return sorted(list(source_types))
        except Exception as e:
            logger.error(f"Error getting source types: {e}")
            return []

    def is_connected(self) -> bool:
        """Check if Qdrant is connected"""
        try:
            self.client.get_collections()
            return True
        except:
            return False

    def optimize_collection(self) -> Dict[str, Any]:
        """
        Trigger collection optimization for better search performance.

        This should be called after bulk insertions to:
        - Merge small segments
        - Rebuild HNSW index
        - Apply quantization to new vectors
        """
        try:
            # Trigger optimization by temporarily lowering threshold
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=0,  # Force immediate indexing
                )
            )

            # Reset to normal threshold
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=20000,
                )
            )

            return {"status": "optimization_triggered", "collection": self.collection_name}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def delete_by_source_type(self, source_type: str) -> Dict[str, Any]:
        """
        Delete all documents with a specific source type.

        Useful for re-indexing documentation from a specific source.

        Args:
            source_type: The source type to delete

        Returns:
            Dict with operation status
        """
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_type",
                            match=MatchValue(value=source_type),
                        )
                    ]
                ),
            )
            return {"status": "success", "operation_id": str(result.operation_id) if result else None}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton instance
vector_store = VectorStore()
