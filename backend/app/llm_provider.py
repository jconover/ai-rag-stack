"""Multi-provider LLM abstraction layer.

Defines an abstract ``LLMProvider`` with two concrete implementations:

- ``OllamaProvider``: wraps the existing local Ollama calls used by the
  RAG pipeline. Preserves historical behavior (blocking client run in a
  thread, circuit breaker, same options dict) so swapping the provider
  is a no-op when ``LLM_PROVIDER=ollama``.
- ``AnthropicProvider``: uses the official async ``anthropic`` SDK.
  Default model is ``claude-haiku-4-5-20251001``. Supports both
  non-streaming (``messages.create``) and streaming (``messages.stream``).

A factory ``get_llm_provider()`` returns a process-wide singleton based
on ``settings.llm_provider``. Callers that need deterministic behavior
can also instantiate a provider directly.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

import ollama

from app.config import settings
from app.circuit_breaker import ollama_circuit_breaker

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract async LLM provider."""

    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Return a full completion string for the given prompt."""

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Yield completion text chunks as they are produced."""
        # pragma: no cover - abstract; subclasses implement as async generator
        if False:
            yield ""


class OllamaProvider(LLMProvider):
    """Ollama provider wrapping the existing blocking client.

    The upstream ``ollama`` python client is synchronous; we run the call
    in ``asyncio.to_thread`` so the event loop is not blocked. This keeps
    behavior identical to the existing ``rag.py`` pipeline, including the
    circuit breaker wrapper.
    """

    name = "ollama"

    def __init__(self, model: Optional[str] = None):
        self.model = model or settings.ollama_default_model

    def _build_messages(self, prompt: str, system: Optional[str]):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        messages = self._build_messages(prompt, system)
        model = self.model

        def _call():
            return ollama_circuit_breaker.call(
                lambda: ollama.chat(
                    model=model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                )
            )

        response = await asyncio.to_thread(_call)
        return response["message"]["content"]

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        messages = self._build_messages(prompt, system)
        model = self.model
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        _sentinel = object()

        def _worker():
            try:
                stream = ollama_circuit_breaker.call(
                    lambda: ollama.chat(
                        model=model,
                        messages=messages,
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                        stream=True,
                    )
                )
                for chunk in stream:
                    content = chunk.get("message", {}).get("content")
                    if content:
                        loop.call_soon_threadsafe(queue.put_nowait, content)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _sentinel)

        task = loop.run_in_executor(None, _worker)
        try:
            while True:
                item = await queue.get()
                if item is _sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            await task


class AnthropicProvider(LLMProvider):
    """Anthropic provider using the async ``anthropic`` SDK.

    Uses ``AsyncAnthropic`` with ``messages.create`` for non-stream and
    ``messages.stream`` for incremental token delivery.
    """

    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "anthropic package is not installed. Add `anthropic>=0.40.0` to requirements.txt "
                "and rebuild the backend image."
            ) from exc

        self.model = model or settings.anthropic_model
        key = api_key or settings.anthropic_api_key
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; cannot use AnthropicProvider."
            )
        self._client = AsyncAnthropic(api_key=key)

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        msg = await self._client.messages.create(**kwargs)

        # Concatenate all text blocks
        parts = []
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_provider_cache: dict = {}


def get_llm_provider(name: Optional[str] = None) -> LLMProvider:
    """Return a cached ``LLMProvider`` for the given provider name.

    If ``name`` is None, uses ``settings.llm_provider`` (default: ``ollama``).
    """
    provider_name = (name or settings.llm_provider or "ollama").lower()
    if provider_name in _provider_cache:
        return _provider_cache[provider_name]

    if provider_name == "ollama":
        provider: LLMProvider = OllamaProvider()
    elif provider_name == "anthropic":
        provider = AnthropicProvider()
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider_name}'. Expected 'ollama' or 'anthropic'."
        )

    _provider_cache[provider_name] = provider
    logger.info(
        "Initialized LLM provider: %s (model=%s)",
        provider.name,
        getattr(provider, "model", "?"),
    )
    return provider


def reset_provider_cache() -> None:
    """Clear the provider cache (primarily for tests)."""
    _provider_cache.clear()
