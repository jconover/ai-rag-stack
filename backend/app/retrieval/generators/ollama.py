"""Ollama-based response generator.

Wraps Ollama LLM calls for the RAG pipeline with support for streaming.
"""

import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, List

from app.retrieval.generators.base import ResponseGenerator, GenerationResult
from app.config import settings

logger = logging.getLogger(__name__)


class OllamaGenerator(ResponseGenerator):
    """Response generator using Ollama local LLM.

    Supports both synchronous and streaming generation with
    model-specific prompt formatting.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy load Ollama client."""
        if self._client is None:
            import ollama
            self._client = ollama
        return self._client

    @property
    def name(self) -> str:
        return "ollama"

    def _build_system_prompt(self, model: str, query: str) -> str:
        """Build system prompt based on model type."""
        base_prompt = """You are a helpful DevOps AI assistant with expertise in:
- Kubernetes, Docker, and container orchestration
- CI/CD pipelines (GitHub Actions, Jenkins, GitLab CI)
- Infrastructure as Code (Terraform, Ansible, Pulumi)
- Cloud platforms (AWS, GCP, Azure)
- Monitoring and observability (Prometheus, Grafana)

Answer questions accurately using the provided context. If the context doesn't
contain enough information, say so. Include code examples when helpful."""

        return base_prompt

    def _build_messages(
        self,
        query: str,
        context: str,
        model: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Build the message list for Ollama."""
        messages = []

        # System prompt
        system_prompt = self._build_system_prompt(model, query)
        messages.append({"role": "system", "content": system_prompt})

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        # User message with context
        user_content = f"""Context from documentation:
{context}

Question: {query}

Please provide a helpful answer based on the context above."""

        messages.append({"role": "user", "content": user_content})

        return messages

    async def generate(
        self,
        query: str,
        context: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> GenerationResult:
        """Generate a response using Ollama."""
        import asyncio

        model = model or settings.ollama_default_model
        start_time = time.perf_counter()

        conversation_history = kwargs.get('conversation_history')
        messages = self._build_messages(query, context, model, conversation_history)

        try:
            client = self._get_client()
            loop = asyncio.get_running_loop()

            response = await loop.run_in_executor(
                None,
                lambda: client.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                )
            )

            generation_time_ms = (time.perf_counter() - start_time) * 1000
            response_text = response.get('message', {}).get('content', '')

            return GenerationResult(
                response=response_text,
                model=model,
                generation_time_ms=generation_time_ms,
                generator_name=self.name,
                metadata={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return GenerationResult(
                response="",
                model=model,
                generation_time_ms=(time.perf_counter() - start_time) * 1000,
                generator_name=self.name,
                error=str(e),
            )

    async def generate_stream(
        self,
        query: str,
        context: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate a streaming response using Ollama."""
        import asyncio

        model = model or settings.ollama_default_model
        start_time = time.perf_counter()

        conversation_history = kwargs.get('conversation_history')
        messages = self._build_messages(query, context, model, conversation_history)

        # Yield metadata first
        yield {
            "type": "metadata",
            "model": model,
            "temperature": temperature,
        }

        try:
            client = self._get_client()
            loop = asyncio.get_running_loop()

            # Run streaming in executor
            def stream_sync():
                return client.chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                )

            stream = await loop.run_in_executor(None, stream_sync)

            full_response = ""
            for chunk in stream:
                content = chunk.get('message', {}).get('content', '')
                if content:
                    full_response += content
                    yield {
                        "type": "content",
                        "content": content,
                    }

            generation_time_ms = (time.perf_counter() - start_time) * 1000

            yield {
                "type": "done",
                "generation_time_ms": generation_time_ms,
                "total_length": len(full_response),
            }

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield {
                "type": "error",
                "error": str(e),
            }

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ollama_host": settings.ollama_host,
            "default_model": settings.ollama_default_model,
        }

    def is_available(self) -> bool:
        try:
            client = self._get_client()
            client.list()
            return True
        except Exception:
            return False


__all__ = ["OllamaGenerator"]
