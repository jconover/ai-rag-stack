"""Cross-encoder reranker for improved RAG retrieval quality.

This module implements a cross-encoder reranking stage that scores query-document
pairs more accurately than bi-encoder similarity search alone. Cross-encoders
jointly encode the query and document, enabling richer interaction between them.

Typical usage:
    1. Retrieve top-K candidates using fast bi-encoder (vector search)
    2. Rerank candidates using cross-encoder for higher precision
    3. Return top-N reranked results to the LLM
"""

import logging
from typing import List, Optional, Tuple

import torch
from sentence_transformers import CrossEncoder

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from app.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for improving retrieval quality.

    Uses a cross-encoder model to score query-document pairs more accurately
    than bi-encoder similarity. This is computationally more expensive but
    yields better relevance ranking.

    Attributes:
        model: The cross-encoder model instance
        device: The device (cpu/cuda) the model runs on
        model_name: Name of the loaded model
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """Initialize the reranker with a cross-encoder model.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
                       Defaults to settings.reranker_model.
            device: Device to run inference on ('cpu' or 'cuda').
                   Defaults to settings.reranker_device.
            batch_size: Batch size for processing multiple query-document pairs.
        """
        self.model_name = model_name or settings.reranker_model
        self.batch_size = batch_size

        # Determine device
        requested_device = device or settings.reranker_device
        if requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available, falling back to CPU"
            )
            self.device = "cpu"
        else:
            self.device = requested_device

        logger.info(
            f"Loading cross-encoder model '{self.model_name}' on device '{self.device}'"
        )

        # Initialize the cross-encoder model
        self.model = CrossEncoder(
            self.model_name,
            max_length=512,
            device=self.device,
        )

        # Warmup the model to eliminate first-query cold-start latency
        self._warmup()

        logger.info(f"Reranker initialized successfully on {self.device}")

    def _warmup(self) -> None:
        """Warmup the model with a dummy query to initialize all components.

        This eliminates the cold-start latency on the first real query by
        forcing model initialization, JIT compilation, and memory allocation.
        """
        logger.debug("Warming up reranker model...")
        dummy_pairs = [("warmup query", "warmup document")]
        try:
            self.model.predict(dummy_pairs)
            logger.debug("Reranker warmup complete")
        except Exception as e:
            logger.warning(f"Reranker warmup failed (non-critical): {e}")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """Rerank documents by relevance to the query using cross-encoder scoring.

        Args:
            query: The search query string
            documents: List of Document objects to rerank
            top_k: Number of top documents to return after reranking.
                  Defaults to settings.reranker_top_k.

        Returns:
            List of Document objects sorted by relevance score (highest first),
            truncated to top_k results. Each document's metadata is updated
            with 'rerank_score' containing the cross-encoder score.

        Note:
            If documents is empty or top_k <= 0, returns an empty list.
            If top_k > len(documents), returns all documents reranked.
        """
        if not documents:
            return []

        if top_k is None:
            top_k = settings.reranker_top_k

        if top_k <= 0:
            return []

        # Create query-document pairs for scoring
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs using the cross-encoder
        # The model handles batching internally based on batch_size
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original documents unchanged on error
            return documents[:top_k]

        # Combine documents with their scores
        scored_docs: List[Tuple[float, Document]] = list(zip(scores, documents))

        # Sort by score descending (highest relevance first)
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Extract top-k documents and update metadata with rerank scores
        reranked_docs = []
        for score, doc in scored_docs[:top_k]:
            # Create a copy of metadata to avoid mutating original
            updated_metadata = doc.metadata.copy()
            updated_metadata['rerank_score'] = float(score)

            # Create new Document with updated metadata
            reranked_doc = Document(
                page_content=doc.page_content,
                metadata=updated_metadata,
            )
            reranked_docs.append(reranked_doc)

        logger.debug(
            f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}"
        )

        return reranked_docs

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Rerank documents and return them with their scores.

        This is a convenience method when you need explicit access to scores
        without checking metadata.

        Args:
            query: The search query string
            documents: List of Document objects to rerank
            top_k: Number of top documents to return after reranking.

        Returns:
            List of (Document, score) tuples sorted by score descending.
        """
        reranked = self.rerank(query, documents, top_k)
        return [
            (doc, doc.metadata.get('rerank_score', 0.0))
            for doc in reranked
        ]

    def score_pairs(
        self,
        query: str,
        texts: List[str],
    ) -> List[float]:
        """Score query-text pairs without Document wrapping.

        Useful for scoring arbitrary text snippets against a query.

        Args:
            query: The search query string
            texts: List of text strings to score against the query

        Returns:
            List of relevance scores (higher = more relevant)
        """
        if not texts:
            return []

        pairs = [(query, text) for text in texts]

        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            return [float(s) for s in scores]
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return [0.0] * len(texts)

    def get_model_info(self) -> dict:
        """Get information about the loaded reranker model.

        Returns:
            Dictionary with model name, device, and max sequence length.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.model.max_length,
            "batch_size": self.batch_size,
        }


# Lazy singleton instance - only initialized if reranker is enabled
_reranker_instance: Optional[Reranker] = None


def get_reranker() -> Optional[Reranker]:
    """Get the singleton reranker instance.

    Returns None if reranking is disabled in settings.
    Initializes the reranker on first call if enabled.

    Returns:
        Reranker instance or None if disabled.
    """
    global _reranker_instance

    if not settings.reranker_enabled:
        return None

    if _reranker_instance is None:
        _reranker_instance = Reranker()

    return _reranker_instance


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None,
) -> List[Document]:
    """Convenience function to rerank documents if reranker is enabled.

    If reranking is disabled, returns the original documents truncated to top_k.

    Args:
        query: The search query string
        documents: List of Document objects to rerank
        top_k: Number of top documents to return. Defaults to settings.reranker_top_k
               if reranker is enabled, otherwise settings.top_k_results.

    Returns:
        Reranked documents if reranker is enabled, otherwise original documents.
    """
    reranker = get_reranker()

    if reranker is None:
        # Reranker disabled - return original documents
        if top_k is None:
            top_k = settings.top_k_results
        return documents[:top_k]

    return reranker.rerank(query, documents, top_k)
