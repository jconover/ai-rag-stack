"""Response generators for the RAG pipeline.

This package provides different strategies for generating responses using LLMs.
"""

from app.retrieval.generators.base import ResponseGenerator, GenerationResult
from app.retrieval.generators.ollama import OllamaGenerator

__all__ = ["ResponseGenerator", "GenerationResult", "OllamaGenerator"]
