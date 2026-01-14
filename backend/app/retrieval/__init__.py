"""Modular retrieval pipeline for RAG.

This package provides a modular architecture for the RAG pipeline with:
- Query expanders (HyDE, conversation context)
- Retrieval strategies (hybrid search)
- Response generators (Ollama)
- Pipeline orchestrator

Usage:
    from app.retrieval import RAGPipelineOrchestrator, create_default_orchestrator

    orchestrator = create_default_orchestrator()
    result = await orchestrator.process_query("How do I scale Kubernetes?")
"""

from app.retrieval.base import RetrievalStrategy, RetrievalResult, Document
from app.retrieval.pipeline import RAGPipelineOrchestrator, create_default_orchestrator

__all__ = [
    # Base classes
    "RetrievalStrategy",
    "RetrievalResult",
    "Document",
    # Orchestrator
    "RAGPipelineOrchestrator",
    "create_default_orchestrator",
]
