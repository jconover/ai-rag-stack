"""Device detection and selection utilities for ML models.

Provides automatic GPU detection with graceful fallback to CPU for
embedding models and cross-encoder rerankers.

Supported devices:
- CUDA (NVIDIA GPUs): Best performance for most models
- MPS (Apple Silicon): Native acceleration on M1/M2/M3 Macs
- CPU: Universal fallback

Usage:
    from app.device_utils import get_optimal_device, get_device_info

    # Auto-detect best available device
    device = get_optimal_device()  # Returns "cuda", "mps", or "cpu"

    # Request specific device with fallback
    device = get_optimal_device("cuda")  # Falls back to CPU if no NVIDIA GPU

    # Get detailed device information for health checks
    info = get_device_info()
"""

import logging
from functools import lru_cache
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Cached device availability flags
_cuda_available: Optional[bool] = None
_mps_available: Optional[bool] = None
_torch_available: Optional[bool] = None


def _check_torch_available() -> bool:
    """Check if PyTorch is installed."""
    global _torch_available
    if _torch_available is not None:
        return _torch_available

    try:
        import torch  # noqa: F401
        _torch_available = True
    except ImportError:
        _torch_available = False
        logger.debug("PyTorch not installed, GPU acceleration unavailable")

    return _torch_available


def _check_cuda_available() -> bool:
    """Check if CUDA (NVIDIA GPU) is available."""
    global _cuda_available
    if _cuda_available is not None:
        return _cuda_available

    if not _check_torch_available():
        _cuda_available = False
        return False

    try:
        import torch
        _cuda_available = torch.cuda.is_available()
        if _cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logger.info(f"CUDA available: {device_count} device(s) detected ({device_name})")
        else:
            logger.debug("CUDA not available")
    except Exception as e:
        logger.debug(f"CUDA detection failed: {e}")
        _cuda_available = False

    return _cuda_available


def _check_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    global _mps_available
    if _mps_available is not None:
        return _mps_available

    if not _check_torch_available():
        _mps_available = False
        return False

    try:
        import torch
        # MPS support was added in PyTorch 1.12
        if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
            _mps_available = torch.backends.mps.is_available()
            if _mps_available:
                logger.info("MPS (Apple Silicon) acceleration available")
            else:
                logger.debug("MPS not available")
        else:
            _mps_available = False
            logger.debug("MPS support not available in this PyTorch version")
    except Exception as e:
        logger.debug(f"MPS detection failed: {e}")
        _mps_available = False

    return _mps_available


def get_optimal_device(preferred: str = "auto") -> str:
    """Detect optimal device for ML models with graceful fallback.

    Checks for GPU availability and returns the best available device
    based on the preference setting.

    Args:
        preferred: Device preference - one of:
            - "auto": Automatically select best available (CUDA > MPS > CPU)
            - "cuda": Use NVIDIA GPU if available, else CPU
            - "mps": Use Apple Silicon if available, else CPU
            - "cpu": Always use CPU (no GPU acceleration)

    Returns:
        Device string suitable for PyTorch/sentence-transformers:
        - "cuda": NVIDIA GPU
        - "mps": Apple Silicon GPU
        - "cpu": CPU fallback

    Examples:
        >>> get_optimal_device()  # Auto-detect
        'cuda'  # or 'mps' or 'cpu'

        >>> get_optimal_device("cuda")  # Request CUDA
        'cuda'  # if available, else 'cpu'

        >>> get_optimal_device("cpu")  # Force CPU
        'cpu'
    """
    # Normalize preference
    preferred = preferred.lower().strip() if preferred else "auto"

    # CPU is always available
    if preferred == "cpu":
        logger.debug("CPU device selected (explicitly requested)")
        return "cpu"

    # Try CUDA (NVIDIA)
    if preferred in ("auto", "cuda"):
        if _check_cuda_available():
            logger.debug("CUDA device selected")
            return "cuda"
        elif preferred == "cuda":
            logger.warning("CUDA requested but not available, falling back to CPU")

    # Try MPS (Apple Silicon)
    if preferred in ("auto", "mps"):
        if _check_mps_available():
            logger.debug("MPS device selected")
            return "mps"
        elif preferred == "mps":
            logger.warning("MPS requested but not available, falling back to CPU")

    # Fallback to CPU
    logger.debug("CPU device selected (fallback)")
    return "cpu"


def get_device_info() -> Dict[str, Any]:
    """Get detailed information about available compute devices.

    Returns a dictionary with device availability and details suitable
    for health check endpoints and diagnostics.

    Returns:
        Dictionary with device information:
        {
            "pytorch_available": bool,
            "cuda_available": bool,
            "cuda_device_count": int,
            "cuda_device_name": str or None,
            "cuda_memory_total_gb": float or None,
            "cuda_memory_free_gb": float or None,
            "mps_available": bool,
            "recommended_device": str,
            "current_embedding_device": str,
            "current_reranker_device": str,
        }
    """
    from app.config import settings

    info: Dict[str, Any] = {
        "pytorch_available": _check_torch_available(),
        "cuda_available": _check_cuda_available(),
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "cuda_memory_total_gb": None,
        "cuda_memory_free_gb": None,
        "mps_available": _check_mps_available(),
        "recommended_device": get_optimal_device("auto"),
        "current_embedding_device": settings.embedding_device,
        "current_reranker_device": settings.reranker_device,
    }

    # Get CUDA details if available
    if info["cuda_available"]:
        try:
            import torch
            info["cuda_device_count"] = torch.cuda.device_count()
            if info["cuda_device_count"] > 0:
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
                # Get memory info for first device
                memory_info = torch.cuda.mem_get_info(0)
                info["cuda_memory_free_gb"] = round(memory_info[0] / (1024**3), 2)
                info["cuda_memory_total_gb"] = round(memory_info[1] / (1024**3), 2)
        except Exception as e:
            logger.debug(f"Failed to get CUDA details: {e}")

    return info


def log_device_configuration() -> None:
    """Log the current device configuration at startup.

    Call this during application startup to log which devices are
    being used for embeddings and reranking.
    """
    from app.config import settings

    info = get_device_info()

    logger.info("=" * 60)
    logger.info("ML Device Configuration")
    logger.info("=" * 60)
    logger.info(f"PyTorch available: {info['pytorch_available']}")
    logger.info(f"CUDA available: {info['cuda_available']}")
    if info['cuda_available']:
        logger.info(f"  - Device count: {info['cuda_device_count']}")
        logger.info(f"  - Device name: {info['cuda_device_name']}")
        if info['cuda_memory_total_gb']:
            logger.info(f"  - Memory: {info['cuda_memory_free_gb']:.1f} GB free / {info['cuda_memory_total_gb']:.1f} GB total")
    logger.info(f"MPS available: {info['mps_available']}")
    logger.info(f"Recommended device: {info['recommended_device']}")
    logger.info("-" * 60)
    logger.info(f"Embedding device (configured): {settings.embedding_device}")
    logger.info(f"Reranker device (configured): {settings.reranker_device}")

    # Warn if configured device doesn't match recommended
    embedding_actual = get_optimal_device(settings.embedding_device)
    reranker_actual = get_optimal_device(settings.reranker_device)

    if settings.embedding_device != "auto" and embedding_actual != settings.embedding_device:
        logger.warning(
            f"Embedding device '{settings.embedding_device}' not available, "
            f"using '{embedding_actual}' instead"
        )

    if settings.reranker_enabled:
        if settings.reranker_device != "auto" and reranker_actual != settings.reranker_device:
            logger.warning(
                f"Reranker device '{settings.reranker_device}' not available, "
                f"using '{reranker_actual}' instead"
            )

    logger.info("=" * 60)


@lru_cache(maxsize=1)
def get_actual_embedding_device() -> str:
    """Get the actual device being used for embeddings.

    Takes the configured preference and resolves it to an actual device,
    with fallback if the preferred device is unavailable.

    Returns:
        Actual device string being used ("cuda", "mps", or "cpu")
    """
    from app.config import settings
    return get_optimal_device(settings.embedding_device)


@lru_cache(maxsize=1)
def get_actual_reranker_device() -> str:
    """Get the actual device being used for the reranker.

    Takes the configured preference and resolves it to an actual device,
    with fallback if the preferred device is unavailable.

    Returns:
        Actual device string being used ("cuda", "mps", or "cpu")
    """
    from app.config import settings
    return get_optimal_device(settings.reranker_device)
