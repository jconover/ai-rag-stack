"""Base classes for response generation strategies.

This module defines the core interfaces for implementing response generators.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result container for response generation operations.

    Attributes:
        response: The generated response text
        model: Model used for generation
        metadata: Additional metadata about the generation
        generation_time_ms: Time taken for generation in milliseconds
        generator_name: Name of the generator that produced this result
        error: Error message if generation failed
        prompt_tokens: Estimated prompt tokens (if available)
        completion_tokens: Estimated completion tokens (if available)
    """
    response: str
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time_ms: float = 0.0
    generator_name: str = ""
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "response": self.response,
            "model": self.model,
            "generation_time_ms": self.generation_time_ms,
            "generator_name": self.generator_name,
            "error": self.error,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "metadata": self.metadata,
        }


class ResponseGenerator(ABC):
    """Abstract base class for response generation strategies.

    Response generators take a query and context and produce an answer
    using an LLM or other generation method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this generator."""
        pass

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> GenerationResult:
        """Generate a response asynchronously.

        Args:
            query: The user's question
            context: Retrieved context to answer from
            model: Model to use for generation
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generator-specific parameters

        Returns:
            GenerationResult containing the response and metadata
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        query: str,
        context: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate a streaming response asynchronously.

        Yields:
            Chunks with type field (metadata, content, done, error)
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration of this generator."""
        return {"name": self.name}

    def is_available(self) -> bool:
        """Check if this generator is available for use."""
        return True


__all__ = ["ResponseGenerator", "GenerationResult"]
