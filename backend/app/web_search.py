"""Web search fallback module using Tavily API.

When local vector search returns low-confidence results, this module
provides web search fallback to find relevant information from the internet.

Usage:
    from app.web_search import web_searcher

    # Check if web search should be used
    if web_searcher.should_search(avg_similarity_score=0.3):
        results = await web_searcher.search("What is AWS WAF")
        # Returns list of WebSearchResult with content, url, score
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"


@dataclass
class WebSearchResult:
    """A single web search result."""
    title: str
    url: str
    content: str
    score: float = 0.0
    raw_content: Optional[str] = None


@dataclass
class WebSearchResponse:
    """Response from web search."""
    query: str
    results: List[WebSearchResult] = field(default_factory=list)
    search_time_ms: float = 0.0
    error: Optional[str] = None
    triggered: bool = False
    trigger_reason: Optional[str] = None


class TavilySearcher:
    """Web search using Tavily API for RAG fallback.

    Tavily is designed for AI/RAG applications and provides:
    - Pre-extracted, cleaned content ready for LLM context
    - Relevance scoring for retrieved pages
    - Domain filtering (include/exclude specific sites)

    Example:
        searcher = TavilySearcher()

        # Search with domain filter
        response = await searcher.search(
            "What is AWS WAF",
            include_domains=["docs.aws.amazon.com"]
        )

        for result in response.results:
            print(f"{result.title}: {result.content[:200]}...")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_results: Optional[int] = None,
    ):
        """Initialize the Tavily searcher.

        Args:
            api_key: Tavily API key. If not provided, reads from settings.
            timeout_seconds: Request timeout. If not provided, reads from settings.
            max_results: Maximum results to return. If not provided, reads from settings.
        """
        self.api_key = api_key or settings.web_search_api_key
        self.timeout_seconds = timeout_seconds or settings.web_search_timeout_seconds
        self.max_results = max_results or settings.web_search_max_results
        self.min_score_threshold = settings.web_search_min_score_threshold

        # Parse domain filters from settings
        self._include_domains = self._parse_domains(settings.web_search_include_domains)
        self._exclude_domains = self._parse_domains(settings.web_search_exclude_domains)

    @staticmethod
    def _parse_domains(domains_str: str) -> List[str]:
        """Parse comma-separated domain string into list."""
        if not domains_str:
            return []
        return [d.strip() for d in domains_str.split(",") if d.strip()]

    @property
    def enabled(self) -> bool:
        """Check if web search is enabled and configured."""
        return settings.web_search_enabled and bool(self.api_key)

    def should_search(
        self,
        avg_similarity_score: float,
        max_similarity_score: Optional[float] = None,
        result_count: int = 0,
    ) -> tuple[bool, Optional[str]]:
        """Determine if web search fallback should be triggered.

        Args:
            avg_similarity_score: Average similarity score from local retrieval
            max_similarity_score: Maximum similarity score (optional)
            result_count: Number of local results returned

        Returns:
            Tuple of (should_search: bool, reason: Optional[str])
        """
        if not self.enabled:
            return False, None

        # No results at all
        if result_count == 0:
            return True, "no_local_results"

        # Low average score
        if avg_similarity_score < self.min_score_threshold:
            return True, f"low_avg_score_{avg_similarity_score:.3f}"

        # Low max score (if provided)
        if max_similarity_score is not None and max_similarity_score < self.min_score_threshold:
            return True, f"low_max_score_{max_similarity_score:.3f}"

        return False, None

    async def search(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: str = "basic",
        include_raw_content: bool = False,
    ) -> WebSearchResponse:
        """Search the web using Tavily API.

        Args:
            query: Search query
            include_domains: Only include results from these domains
            exclude_domains: Exclude results from these domains
            search_depth: "basic" (1 credit) or "advanced" (2 credits)
            include_raw_content: Include full page content (more tokens)

        Returns:
            WebSearchResponse with results or error
        """
        response = WebSearchResponse(query=query, triggered=True)

        if not self.enabled:
            response.error = "Web search not enabled or API key not configured"
            response.triggered = False
            return response

        start_time = time.perf_counter()

        # Merge instance domains with call-specific domains
        final_include = include_domains or self._include_domains
        final_exclude = exclude_domains or self._exclude_domains

        payload: Dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": self.max_results,
            "include_raw_content": include_raw_content,
        }

        if final_include:
            payload["include_domains"] = final_include
        if final_exclude:
            payload["exclude_domains"] = final_exclude

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                http_response = await client.post(TAVILY_API_URL, json=payload)
                http_response.raise_for_status()
                data = http_response.json()

            # Parse results
            for item in data.get("results", []):
                result = WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    raw_content=item.get("raw_content") if include_raw_content else None,
                )
                response.results.append(result)

            logger.info(
                f"Tavily search completed: query='{query[:50]}...' "
                f"results={len(response.results)} time={time.perf_counter() - start_time:.2f}s"
            )

        except httpx.TimeoutException:
            response.error = f"Tavily search timed out after {self.timeout_seconds}s"
            logger.warning(response.error)

        except httpx.HTTPStatusError as e:
            response.error = f"Tavily API error: {e.response.status_code} - {e.response.text[:200]}"
            logger.error(response.error)

        except Exception as e:
            response.error = f"Web search failed: {str(e)}"
            logger.error(response.error)

        response.search_time_ms = (time.perf_counter() - start_time) * 1000
        return response

    def search_sync(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: str = "basic",
    ) -> WebSearchResponse:
        """Synchronous wrapper for search.

        Use this when not in an async context.
        """
        return asyncio.run(
            self.search(
                query=query,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                search_depth=search_depth,
            )
        )

    def format_for_context(
        self,
        results: List[WebSearchResult],
        max_chars_per_result: int = 1500,
    ) -> str:
        """Format web search results for LLM context.

        Args:
            results: List of WebSearchResult
            max_chars_per_result: Maximum characters per result

        Returns:
            Formatted string for LLM context
        """
        if not results:
            return ""

        formatted_parts = []
        for i, result in enumerate(results, 1):
            content = result.content[:max_chars_per_result]
            if len(result.content) > max_chars_per_result:
                content += "..."

            formatted_parts.append(
                f"[Web Source {i}] {result.title}\n"
                f"URL: {result.url}\n"
                f"{content}"
            )

        return "\n\n".join(formatted_parts)

    def get_status(self) -> Dict[str, Any]:
        """Get current status of web search configuration."""
        return {
            "enabled": self.enabled,
            "api_key_configured": bool(self.api_key),
            "min_score_threshold": self.min_score_threshold,
            "max_results": self.max_results,
            "timeout_seconds": self.timeout_seconds,
            "include_domains": self._include_domains,
            "exclude_domains": self._exclude_domains,
        }


# Singleton instance
web_searcher = TavilySearcher()


# Convenience functions
async def search_web(query: str, **kwargs) -> WebSearchResponse:
    """Convenience function to search using singleton."""
    return await web_searcher.search(query, **kwargs)


def should_use_web_search(
    avg_similarity_score: float,
    max_similarity_score: Optional[float] = None,
    result_count: int = 0,
) -> tuple[bool, Optional[str]]:
    """Convenience function to check if web search should trigger."""
    return web_searcher.should_search(
        avg_similarity_score=avg_similarity_score,
        max_similarity_score=max_similarity_score,
        result_count=result_count,
    )
