"""Cross-encoder reranker for improved RAG retrieval quality.

This module implements a cross-encoder reranking stage that scores query-document
pairs more accurately than bi-encoder similarity search alone. Cross-encoders
jointly encode the query and document, enabling richer interaction between them.

Typical usage:
    1. Retrieve top-K candidates using fast bi-encoder (vector search)
    2. Rerank candidates using cross-encoder for higher precision
    3. Return top-N reranked results to the LLM

Platt Scaling Calibration:
    Raw cross-encoder scores can be calibrated to probabilities using Platt scaling.
    This is useful for:
    - Interpretable confidence scores (0-1 probability of relevance)
    - Consistent thresholding across queries
    - Better score fusion when combining multiple rankers

    Enable with RERANKER_CALIBRATION_ENABLED=true and set calibration parameters
    RERANKER_CALIBRATION_A and RERANKER_CALIBRATION_B (learned from labeled data).
"""

import json
import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.special import expit  # sigmoid function
from sentence_transformers import CrossEncoder

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from app.config import settings
from app.device_utils import get_optimal_device, get_actual_reranker_device

logger = logging.getLogger(__name__)


class ScoreCalibrator:
    """Platt scaling calibrator for converting raw reranker scores to probabilities.

    Platt scaling fits a logistic regression model to map raw classifier/ranker
    outputs to calibrated probabilities. The formula is:

        P(relevant) = sigmoid(a * score + b) = 1 / (1 + exp(-(a * score + b)))

    Where:
        - score: raw cross-encoder output
        - a, b: learned calibration parameters
        - P(relevant): calibrated probability of relevance

    Attributes:
        a: Slope parameter for Platt scaling (default: 1.0)
        b: Intercept parameter for Platt scaling (default: 0.0)

    Example usage:
        # Initialize with default or configured parameters
        calibrator = ScoreCalibrator(a=2.5, b=-1.2)

        # Calibrate a single score
        prob = calibrator.calibrate(0.75)  # Returns ~0.85

        # Calibrate a batch of scores
        probs = calibrator.calibrate_batch([0.2, 0.5, 0.8])

        # Fit from labeled data
        calibrator = ScoreCalibrator.fit(
            scores=[0.1, 0.3, 0.7, 0.9],
            labels=[0, 0, 1, 1]
        )
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        """Initialize the calibrator with Platt scaling parameters.

        Args:
            a: Slope parameter. Higher values increase the steepness of the
               sigmoid, making the calibration more sensitive to score changes.
               Default 1.0 applies minimal transformation.
            b: Intercept parameter. Shifts the sigmoid left (positive b) or
               right (negative b), affecting the threshold at which scores
               map to 0.5 probability. Default 0.0 centers at score=0.
        """
        self.a = a
        self.b = b

    def calibrate(self, raw_score: float) -> float:
        """Convert a raw cross-encoder score to a calibrated probability.

        Args:
            raw_score: Raw score from the cross-encoder model. Typically
                      ranges from -10 to +10 for ms-marco models.

        Returns:
            Calibrated probability in [0, 1] representing the likelihood
            that the document is relevant to the query.
        """
        return float(expit(self.a * raw_score + self.b))

    def calibrate_batch(self, scores: List[float]) -> List[float]:
        """Calibrate a batch of scores efficiently using vectorized operations.

        Args:
            scores: List of raw cross-encoder scores.

        Returns:
            List of calibrated probabilities in [0, 1].
        """
        if not scores:
            return []
        arr = np.array(scores, dtype=np.float64)
        calibrated = expit(self.a * arr + self.b)
        return calibrated.tolist()

    @classmethod
    def fit(
        cls,
        scores: List[float],
        labels: List[int],
        regularization: float = 1.0,
    ) -> "ScoreCalibrator":
        """Fit calibration parameters from labeled relevance data.

        Uses logistic regression to learn optimal a and b parameters that
        map raw scores to probability of relevance.

        Args:
            scores: Raw cross-encoder scores from the reranker.
            labels: Binary relevance labels (0 = not relevant, 1 = relevant).
            regularization: Inverse of regularization strength (C parameter
                          in sklearn). Lower values = stronger regularization.
                          Default 1.0 provides moderate regularization.

        Returns:
            ScoreCalibrator instance with fitted parameters.

        Raises:
            ValueError: If scores and labels have different lengths or
                       if labels are not binary (0 or 1).

        Example:
            # Collect labeled examples from user feedback
            scores = [0.1, 0.25, 0.4, 0.6, 0.8, 0.95]
            labels = [0, 0, 0, 1, 1, 1]

            # Fit calibrator
            calibrator = ScoreCalibrator.fit(scores, labels)
            print(f"Fitted: a={calibrator.a:.3f}, b={calibrator.b:.3f}")

            # Save parameters to config
            # RERANKER_CALIBRATION_A={calibrator.a}
            # RERANKER_CALIBRATION_B={calibrator.b}
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError(
                "sklearn is required for fitting calibration parameters. "
                "Install with: pip install scikit-learn"
            )

        if len(scores) != len(labels):
            raise ValueError(
                f"scores and labels must have same length: "
                f"{len(scores)} vs {len(labels)}"
            )

        if not all(label in (0, 1) for label in labels):
            raise ValueError("labels must be binary (0 or 1)")

        if len(set(labels)) < 2:
            raise ValueError(
                "labels must contain at least one positive and one negative example"
            )

        X = np.array(scores, dtype=np.float64).reshape(-1, 1)
        y = np.array(labels, dtype=np.int32)

        lr = LogisticRegression(C=regularization, solver="lbfgs", max_iter=1000)
        lr.fit(X, y)

        a = float(lr.coef_[0][0])
        b = float(lr.intercept_[0])

        logger.info(f"Fitted Platt scaling parameters: a={a:.4f}, b={b:.4f}")

        return cls(a=a, b=b)

    def save(self, path: str) -> None:
        """Save calibration parameters to a JSON file.

        Args:
            path: File path to save parameters to.
        """
        params = {"a": self.a, "b": self.b}
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
        logger.info(f"Saved calibration parameters to {path}")

    @classmethod
    def load(cls, path: str) -> "ScoreCalibrator":
        """Load calibration parameters from a JSON file.

        Args:
            path: File path to load parameters from.

        Returns:
            ScoreCalibrator instance with loaded parameters.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(path, "r") as f:
            params = json.load(f)

        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.0))

        logger.info(f"Loaded calibration parameters from {path}: a={a:.4f}, b={b:.4f}")

        return cls(a=a, b=b)

    def get_params(self) -> dict:
        """Get calibration parameters as a dictionary.

        Returns:
            Dictionary with 'a' and 'b' parameters.
        """
        return {"a": self.a, "b": self.b}

    def __repr__(self) -> str:
        return f"ScoreCalibrator(a={self.a:.4f}, b={self.b:.4f})"


class Reranker:
    """Cross-encoder reranker for improving retrieval quality.

    Uses a cross-encoder model to score query-document pairs more accurately
    than bi-encoder similarity. This is computationally more expensive but
    yields better relevance ranking.

    Optionally applies Platt scaling calibration to convert raw scores to
    calibrated probabilities when RERANKER_CALIBRATION_ENABLED=true.

    Attributes:
        model: The cross-encoder model instance
        device: The device (cpu/cuda) the model runs on
        model_name: Name of the loaded model
        calibrator: Optional ScoreCalibrator for Platt scaling
        calibration_enabled: Whether calibration is active
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: Optional[int] = None,
        calibrator: Optional[ScoreCalibrator] = None,
    ):
        """Initialize the reranker with a cross-encoder model.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
                       Defaults to settings.reranker_model.
            device: Device to run inference on ('cpu', 'cuda', 'mps', or 'auto').
                   Defaults to settings.reranker_device.
            batch_size: Batch size for processing multiple query-document pairs.
            max_length: Maximum sequence length for the cross-encoder.
                       Defaults to settings.reranker_max_length (1024).
            calibrator: Optional ScoreCalibrator for Platt scaling. If None
                       and calibration is enabled in settings, creates one
                       from configured parameters.
        """
        self.model_name = model_name or settings.reranker_model
        self.batch_size = batch_size
        self.max_length = max_length or settings.reranker_max_length

        # Determine device using auto-detection with graceful fallback
        requested_device = device or settings.reranker_device
        self.device = get_optimal_device(requested_device)

        if requested_device not in ("auto", "cpu") and self.device != requested_device:
            logger.warning(
                f"Reranker device '{requested_device}' not available, "
                f"falling back to '{self.device}'"
            )

        logger.info(
            f"Loading cross-encoder model '{self.model_name}' on device '{self.device}' "
            f"(configured: {requested_device}, max_length: {self.max_length})"
        )

        # Initialize the cross-encoder model
        self.model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=self.device,
        )

        # Initialize Platt scaling calibrator
        self.calibration_enabled = settings.reranker_calibration_enabled
        if calibrator is not None:
            self.calibrator = calibrator
        elif self.calibration_enabled:
            self.calibrator = ScoreCalibrator(
                a=settings.reranker_calibration_a,
                b=settings.reranker_calibration_b,
            )
            logger.info(
                f"Platt scaling calibration enabled: a={self.calibrator.a:.4f}, "
                f"b={self.calibrator.b:.4f}"
            )
        else:
            self.calibrator = None

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

    def _check_length(self, query: str, doc: str) -> None:
        """Check if query-document pair may exceed max_length and log warning.

        Cross-encoders tokenize the query and document together with special tokens.
        When the combined length exceeds max_length, the input is truncated which
        may reduce reranking quality for longer DevOps documentation chunks.

        Args:
            query: The search query string
            doc: The document content string
        """
        combined_len = len(query) + len(doc)
        # Rough estimate: ~4 characters per token for English text
        # This is approximate since actual tokenization varies by model
        estimated_tokens = combined_len // 4
        if estimated_tokens > self.max_length:
            logger.warning(
                f"Document may be truncated: ~{estimated_tokens} estimated tokens "
                f"exceeds max_length={self.max_length}. Consider increasing "
                f"RERANKER_MAX_LENGTH or reducing chunk size."
            )

    def _apply_calibration(self, scores: List[float]) -> List[float]:
        """Apply Platt scaling calibration to raw scores if enabled.

        Args:
            scores: List of raw cross-encoder scores.

        Returns:
            Calibrated scores if calibration is enabled, otherwise original scores.
        """
        if self.calibration_enabled and self.calibrator is not None:
            return self.calibrator.calibrate_batch(scores)
        return scores

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
            truncated to top_k results. Each document's metadata is updated with:
            - 'rerank_score': the final score (calibrated if enabled)
            - 'rerank_score_raw': original cross-encoder score (if calibration enabled)
            - 'rerank_calibrated': boolean indicating if score was calibrated

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

        # Check for potential truncation on longer documents
        for doc in documents:
            self._check_length(query, doc.page_content)

        # Score all pairs using the cross-encoder
        # The model handles batching internally based on batch_size
        try:
            raw_scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            raw_scores = [float(s) for s in raw_scores]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original documents unchanged on error
            return documents[:top_k]

        # Apply Platt scaling calibration if enabled
        scores = self._apply_calibration(raw_scores)

        # Combine documents with their scores (using calibrated scores for ranking)
        scored_docs: List[Tuple[float, float, Document]] = list(
            zip(scores, raw_scores, documents)
        )

        # Sort by score descending (highest relevance first)
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Extract top-k documents and update metadata with rerank scores
        reranked_docs = []
        for score, raw_score, doc in scored_docs[:top_k]:
            # Create a copy of metadata to avoid mutating original
            updated_metadata = doc.metadata.copy()
            updated_metadata["rerank_score"] = score
            updated_metadata["rerank_calibrated"] = self.calibration_enabled

            # Include raw score for transparency when calibration is enabled
            if self.calibration_enabled:
                updated_metadata["rerank_score_raw"] = raw_score

            # Create new Document with updated metadata
            reranked_doc = Document(
                page_content=doc.page_content,
                metadata=updated_metadata,
            )
            reranked_docs.append(reranked_doc)

        logger.debug(
            f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}"
            + (f" (calibrated)" if self.calibration_enabled else "")
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
            Scores are calibrated if calibration is enabled.
        """
        reranked = self.rerank(query, documents, top_k)
        return [(doc, doc.metadata.get("rerank_score", 0.0)) for doc in reranked]

    def score_pairs(
        self,
        query: str,
        texts: List[str],
        apply_calibration: bool = True,
    ) -> List[float]:
        """Score query-text pairs without Document wrapping.

        Useful for scoring arbitrary text snippets against a query.

        Args:
            query: The search query string
            texts: List of text strings to score against the query
            apply_calibration: Whether to apply Platt scaling calibration
                              if enabled. Default True.

        Returns:
            List of relevance scores (higher = more relevant).
            Scores are calibrated probabilities in [0, 1] if calibration
            is enabled and apply_calibration is True.
        """
        if not texts:
            return []

        pairs = [(query, text) for text in texts]

        try:
            raw_scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            scores = [float(s) for s in raw_scores]

            if apply_calibration:
                scores = self._apply_calibration(scores)

            return scores
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return [0.0] * len(texts)

    def score_pairs_raw(
        self,
        query: str,
        texts: List[str],
    ) -> List[float]:
        """Score query-text pairs and return raw (uncalibrated) scores.

        Convenience method that always returns raw cross-encoder scores
        regardless of calibration settings.

        Args:
            query: The search query string
            texts: List of text strings to score against the query

        Returns:
            List of raw relevance scores from the cross-encoder.
        """
        return self.score_pairs(query, texts, apply_calibration=False)

    def get_model_info(self) -> dict:
        """Get information about the loaded reranker model.

        Returns:
            Dictionary with model name, device, max sequence length,
            batch size, and calibration settings.
        """
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.model.max_length,
            "batch_size": self.batch_size,
            "calibration_enabled": self.calibration_enabled,
        }

        if self.calibrator is not None:
            info["calibration_params"] = self.calibrator.get_params()

        return info

    def set_calibrator(self, calibrator: ScoreCalibrator) -> None:
        """Set or update the score calibrator.

        Args:
            calibrator: ScoreCalibrator instance to use for Platt scaling.
        """
        self.calibrator = calibrator
        self.calibration_enabled = True
        logger.info(
            f"Updated calibrator: a={calibrator.a:.4f}, b={calibrator.b:.4f}"
        )

    def disable_calibration(self) -> None:
        """Disable Platt scaling calibration."""
        self.calibration_enabled = False
        logger.info("Calibration disabled")

    def enable_calibration(self) -> None:
        """Enable Platt scaling calibration.

        Raises:
            ValueError: If no calibrator has been set.
        """
        if self.calibrator is None:
            raise ValueError(
                "No calibrator set. Use set_calibrator() first or configure "
                "RERANKER_CALIBRATION_A and RERANKER_CALIBRATION_B"
            )
        self.calibration_enabled = True
        logger.info("Calibration enabled")


# Thread-safe singleton for shared reranker model
# This reduces memory usage in multi-worker deployments where each worker
# would otherwise load its own copy of the cross-encoder model (~80MB)
_reranker_instance: Optional[Reranker] = None
_reranker_lock = threading.Lock()


def get_reranker() -> Optional[Reranker]:
    """Get the singleton reranker instance (thread-safe).

    Returns None if reranking is disabled in settings.
    Initializes the reranker on first call if enabled.

    Uses double-checked locking pattern to ensure:
    1. Thread-safety: Only one thread initializes the model
    2. Efficiency: After initialization, no lock acquisition needed
    3. Memory savings: All workers share the same model instance

    Returns:
        Reranker instance or None if disabled.
    """
    global _reranker_instance

    if not settings.reranker_enabled:
        return None

    if _reranker_instance is None:
        with _reranker_lock:
            # Double-check inside lock to prevent race conditions
            if _reranker_instance is None:
                logger.info(
                    f"Initializing shared reranker model: {settings.reranker_model} "
                    f"(device: {settings.reranker_device})"
                )
                _reranker_instance = Reranker()
                logger.info("Shared reranker model initialized successfully")

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


def get_calibrator() -> Optional[ScoreCalibrator]:
    """Get the calibrator from the singleton reranker instance.

    Returns:
        ScoreCalibrator instance or None if reranker is disabled or
        calibration is not configured.
    """
    reranker = get_reranker()
    if reranker is not None:
        return reranker.calibrator
    return None
