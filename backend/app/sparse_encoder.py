"""Sparse vector encoding for hybrid search using BM25.

This module provides sparse vector generation for BM25-based retrieval,
enabling hybrid search that combines semantic (dense) and keyword (sparse) matching.

Uses fastembed's Qdrant/bm25 model for sparse vector generation.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded encoder instance
_sparse_encoder = None


@dataclass
class SparseVector:
    """Sparse vector representation with indices and values."""
    indices: List[int]
    values: List[float]

    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary format for Qdrant."""
        return {
            "indices": self.indices,
            "values": self.values
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List]) -> "SparseVector":
        """Create from dictionary."""
        return cls(
            indices=data.get("indices", []),
            values=data.get("values", [])
        )


class SparseEncoder:
    """BM25-based sparse encoder using fastembed.

    This encoder generates sparse vectors where:
    - Indices represent token IDs in the BM25 vocabulary
    - Values represent BM25 term weights

    These sparse vectors enable keyword-based retrieval that complements
    semantic search, particularly for:
    - Exact term matching
    - Rare/specific terminology
    - Named entities and technical terms
    """

    def __init__(self, model_name: str = None):
        """Initialize the sparse encoder.

        Args:
            model_name: Model name for sparse encoding (default: Qdrant/bm25)
        """
        self.model_name = model_name or settings.sparse_encoder_model
        self._encoder = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of the encoder."""
        if self._initialized:
            return

        try:
            from fastembed import SparseTextEmbedding

            logger.info(f"Initializing sparse encoder: {self.model_name}")
            self._encoder = SparseTextEmbedding(model_name=self.model_name)
            self._initialized = True

            # Warmup the encoder
            list(self._encoder.embed(["warmup"]))
            logger.info("Sparse encoder initialized and warmed up")

        except ImportError:
            logger.error(
                "fastembed not installed. Install with: pip install fastembed"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize sparse encoder: {e}")
            raise

    def encode(self, text: str) -> SparseVector:
        """Encode a single text into a sparse vector.

        Args:
            text: Text to encode

        Returns:
            SparseVector with indices and values
        """
        self._ensure_initialized()

        # fastembed returns a generator, take first result
        embeddings = list(self._encoder.embed([text]))
        if not embeddings:
            return SparseVector(indices=[], values=[])

        embedding = embeddings[0]
        return SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist()
        )

    def encode_batch(self, texts: List[str]) -> List[SparseVector]:
        """Encode multiple texts into sparse vectors.

        Args:
            texts: List of texts to encode

        Returns:
            List of SparseVectors
        """
        if not texts:
            return []

        self._ensure_initialized()

        embeddings = list(self._encoder.embed(texts))
        return [
            SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist()
            )
            for emb in embeddings
        ]

    def encode_query(self, query: str) -> SparseVector:
        """Encode a query for search.

        For BM25, query encoding is the same as document encoding,
        but this method is provided for semantic clarity.

        Args:
            query: Query text to encode

        Returns:
            SparseVector for the query
        """
        return self.encode(query)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "initialized": self._initialized,
            "type": "sparse_bm25"
        }


def get_sparse_encoder() -> Optional[SparseEncoder]:
    """Get the global sparse encoder instance.

    Returns:
        SparseEncoder instance if hybrid search is enabled, None otherwise
    """
    global _sparse_encoder

    if not settings.hybrid_search_enabled:
        return None

    if _sparse_encoder is None:
        _sparse_encoder = SparseEncoder()

    return _sparse_encoder


def encode_for_hybrid(texts: List[str]) -> List[SparseVector]:
    """Encode texts for hybrid search indexing.

    Convenience function for document ingestion.

    Args:
        texts: List of document texts to encode

    Returns:
        List of SparseVectors (empty list if hybrid search disabled)
    """
    encoder = get_sparse_encoder()
    if encoder is None:
        return []

    return encoder.encode_batch(texts)


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Any, float]],
    sparse_results: List[Tuple[Any, float]],
    k: int = None,
    alpha: float = None,
) -> List[Tuple[Any, float]]:
    """Combine dense and sparse search results using Reciprocal Rank Fusion.

    RRF formula: score = sum(1 / (k + rank_i)) for each result list

    This method is robust to score scale differences between dense and sparse
    results because it uses rank positions rather than raw scores.

    Args:
        dense_results: List of (document, score) from dense/vector search
        sparse_results: List of (document, score) from sparse/BM25 search
        k: RRF constant (default from settings, typically 60)
        alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
               Default 0.5 weights both equally

    Returns:
        List of (document, fused_score) sorted by score descending
    """
    if k is None:
        k = settings.hybrid_rrf_k
    if alpha is None:
        alpha = settings.hybrid_search_alpha

    # Build document -> score mapping using content hash as key
    doc_scores: Dict[int, Dict[str, Any]] = {}

    def get_doc_key(doc) -> int:
        """Get unique key for a document based on content."""
        return hash(doc.page_content[:200] if hasattr(doc, 'page_content') else str(doc)[:200])

    # Process dense results
    for rank, (doc, score) in enumerate(dense_results, start=1):
        key = get_doc_key(doc)
        if key not in doc_scores:
            doc_scores[key] = {
                "doc": doc,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_score": 0.0,
                "sparse_score": 0.0,
            }
        doc_scores[key]["dense_rank"] = rank
        doc_scores[key]["dense_score"] = score

    # Process sparse results
    for rank, (doc, score) in enumerate(sparse_results, start=1):
        key = get_doc_key(doc)
        if key not in doc_scores:
            doc_scores[key] = {
                "doc": doc,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_score": 0.0,
                "sparse_score": 0.0,
            }
        doc_scores[key]["sparse_rank"] = rank
        doc_scores[key]["sparse_score"] = score

    # Calculate RRF scores
    fused_results = []
    for key, data in doc_scores.items():
        rrf_dense = 0.0
        rrf_sparse = 0.0

        if data["dense_rank"] is not None:
            rrf_dense = 1.0 / (k + data["dense_rank"])

        if data["sparse_rank"] is not None:
            rrf_sparse = 1.0 / (k + data["sparse_rank"])

        # Weighted combination
        fused_score = alpha * rrf_dense + (1 - alpha) * rrf_sparse

        fused_results.append((data["doc"], fused_score))

    # Sort by fused score descending
    fused_results.sort(key=lambda x: x[1], reverse=True)

    return fused_results
