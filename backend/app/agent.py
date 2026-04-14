"""Agentic RAG loop.

Implements a PLAN -> RETRIEVE -> REFLECT -> DECIDE -> GENERATE -> VERIFY
pipeline on top of the existing RAG building blocks:

- Planning and reflection use the pluggable :class:`LLMProvider`.
- Retrieval reuses ``rag_pipeline._retrieve_with_scores_async`` so HyDE,
  hybrid search, reranking, and the Tavily web-search fallback are all
  preserved without duplication.
- Generation uses ``rag_pipeline._format_context`` +
  ``rag_pipeline._build_messages`` and then calls the provider directly,
  so the final answer flows through the same prompt scaffolding as the
  classic ``/api/chat`` path.

The agent is intentionally pragmatic: LLM calls are kept small and return
JSON so the orchestration remains transparent and debuggable via the
``agent_trace`` returned to callers.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.config import settings
from app.llm_provider import get_llm_provider
from app.rag import rag_pipeline, _split_messages_for_provider, get_model_context_limit

logger = logging.getLogger(__name__)


MAX_RETRIES = 2
REFLECT_SCORE_THRESHOLD = 0.5
MIN_SURVIVING_CHUNKS = 2


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response.

    LLMs often wrap JSON in ```json fences or add surrounding prose. This
    strips common wrappers and falls back to the first ``{...}`` or
    ``[...]`` block it can parse.
    """
    if not text:
        return None
    text = text.strip()
    # Strip triple-backtick fences
    fence = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: find first balanced object/array
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                continue
    return None


class AgenticRAG:
    """Agentic wrapper around the existing RAG pipeline."""

    def __init__(self):
        self.rag = rag_pipeline
        self.provider = get_llm_provider()
        self.model = settings.ollama_default_model

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    async def run(
        self,
        query: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        logger.info("AgenticRAG.run query=%r conversation_id=%s", query, conversation_id)

        # ---- PLAN -------------------------------------------------------
        sub_queries = await self._plan(query)
        trace.append({"step": "plan", "output": sub_queries})
        logger.info("agent.plan -> %s", sub_queries)

        current_query = query
        attempts = 0
        surviving: List[Any] = []  # list of langchain Document objects
        aggregated_meta: List[Dict[str, Any]] = []

        while True:
            surviving = []
            aggregated_meta = []

            # ---- RETRIEVE (per sub-query) -------------------------------
            for sub in sub_queries:
                try:
                    result = await self.rag._retrieve_with_scores_async(
                        sub, self.model, conversation_history=None
                    )
                    docs = result.documents or []
                    trace.append(
                        {
                            "step": "retrieve",
                            "sub_query": sub,
                            "num_chunks": len(docs),
                            "web_search_used": getattr(result, "web_search_used", False),
                        }
                    )
                    logger.info("agent.retrieve sub_query=%r chunks=%d", sub, len(docs))

                    # ---- REFLECT -------------------------------------------
                    kept, dropped = await self._reflect(query, docs)
                    trace.append(
                        {
                            "step": "reflect",
                            "sub_query": sub,
                            "kept": len(kept),
                            "dropped": dropped,
                        }
                    )
                    surviving.extend(kept)
                    aggregated_meta.append(result)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("agent.retrieve failed for sub=%r: %s", sub, exc)
                    trace.append(
                        {"step": "retrieve", "sub_query": sub, "error": str(exc)}
                    )

            # ---- DECIDE -------------------------------------------------
            if len(surviving) >= MIN_SURVIVING_CHUNKS or attempts >= MAX_RETRIES:
                trace.append(
                    {
                        "step": "decide",
                        "action": "proceed",
                        "surviving_chunks": len(surviving),
                        "attempts": attempts,
                    }
                )
                break

            rewritten = await self._rewrite(current_query)
            trace.append(
                {
                    "step": "decide",
                    "action": "retry",
                    "rewritten_query": rewritten,
                    "surviving_chunks": len(surviving),
                    "attempts": attempts,
                }
            )
            logger.info("agent.decide retry rewritten=%r", rewritten)
            sub_queries = [rewritten]
            current_query = rewritten
            attempts += 1

        # ---- GENERATE ---------------------------------------------------
        answer, sources = await self._generate(query, surviving)
        trace.append(
            {
                "step": "generate",
                "answer_length": len(answer),
                "source_count": len(sources),
            }
        )

        # ---- VERIFY -----------------------------------------------------
        verification = await self._verify(query, answer)
        trace.append(
            {
                "step": "verify",
                "addresses_question": verification.get("addresses", False),
                "reason": verification.get("reason", ""),
            }
        )
        logger.info("agent.verify addresses=%s", verification.get("addresses"))

        return {
            "answer": answer,
            "sources": sources,
            "agent_trace": trace,
            "metadata": {
                "retries": attempts,
                "verified": bool(verification.get("addresses", False)),
                "verification_reason": verification.get("reason", ""),
                "conversation_id": conversation_id,
            },
        }

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    async def _plan(self, query: str) -> List[str]:
        system = (
            "You are a planning assistant for a DevOps documentation RAG system. "
            "Given a user question, decide whether it needs ONE retrieval or several "
            "independent sub-queries (e.g. compound questions spanning multiple tools). "
            "Return ONLY valid JSON of the form {\"sub_queries\": [\"...\", \"...\"]}. "
            "Most questions should return a single-element list."
        )
        user = f'User question: "{query}"\n\nReturn JSON only.'
        try:
            raw = await self.provider.complete(
                prompt=user, system=system, temperature=0.0, max_tokens=300
            )
            data = _extract_json(raw)
            if isinstance(data, dict) and isinstance(data.get("sub_queries"), list):
                subs = [s for s in data["sub_queries"] if isinstance(s, str) and s.strip()]
                if subs:
                    return subs[:4]  # cap
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent._plan fallback (%s)", exc)
        return [query]

    async def _reflect(self, original_query: str, docs: List[Any]):
        """Grade chunks 0-1 and keep those >= REFLECT_SCORE_THRESHOLD."""
        if not docs:
            return [], 0

        numbered = []
        for i, d in enumerate(docs, start=1):
            preview = (d.page_content or "").strip().replace("\n", " ")
            numbered.append(f"[{i}] {preview[:500]}")
        chunk_block = "\n\n".join(numbered)

        system = (
            "You grade documentation snippets for relevance to a user question. "
            "Return ONLY JSON mapping the chunk number (as a string) to a float in [0,1]. "
            "Example: {\"1\": 0.9, \"2\": 0.2}. Do not include commentary."
        )
        user = (
            f'Question: "{original_query}"\n\n'
            f"Snippets:\n{chunk_block}\n\n"
            "Return JSON only."
        )

        kept: List[Any] = []
        dropped = 0
        try:
            raw = await self.provider.complete(
                prompt=user, system=system, temperature=0.0, max_tokens=400
            )
            scores = _extract_json(raw) or {}
            if not isinstance(scores, dict):
                scores = {}
            for i, d in enumerate(docs, start=1):
                score = scores.get(str(i))
                try:
                    score = float(score) if score is not None else 0.0
                except (TypeError, ValueError):
                    score = 0.0
                if score >= REFLECT_SCORE_THRESHOLD:
                    # Stash score on metadata for downstream citation ordering
                    d.metadata = dict(d.metadata or {})
                    d.metadata["reflect_score"] = score
                    kept.append(d)
                else:
                    dropped += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent._reflect failed, keeping all docs (%s)", exc)
            kept = list(docs)
            dropped = 0

        # Sort surviving by reflect_score descending
        kept.sort(key=lambda d: d.metadata.get("reflect_score", 0.0), reverse=True)
        return kept, dropped

    async def _rewrite(self, query: str) -> str:
        system = (
            "You rewrite user questions to improve retrieval against DevOps technical "
            "documentation. Return ONLY the rewritten question as a plain string."
        )
        user = f'Original question: "{query}"\n\nRewritten:'
        try:
            raw = await self.provider.complete(
                prompt=user, system=system, temperature=0.2, max_tokens=200
            )
            rewritten = raw.strip().strip('"').strip()
            return rewritten or query
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent._rewrite fallback (%s)", exc)
            return query

    async def _generate(self, query: str, docs: List[Any]):
        """Build context, call the provider, and return (answer, sources)."""
        # Build a citation-aware context block
        context_parts = []
        sources: List[Dict[str, Any]] = []
        for i, d in enumerate(docs, start=1):
            source = d.metadata.get("source", "unknown") if d.metadata else "unknown"
            source_type = d.metadata.get("source_type", "unknown") if d.metadata else "unknown"
            context_parts.append(
                f"[{i}] (source: {source_type} / {source})\n{d.page_content.strip()}"
            )
            sources.append(
                {
                    "index": i,
                    "source": source,
                    "source_type": source_type,
                    "content_preview": (d.page_content or "")[:240],
                    "reflect_score": (d.metadata or {}).get("reflect_score"),
                }
            )
        context_str = "\n\n---\n\n".join(context_parts) if context_parts else ""

        system = (
            "You are a DevOps expert answering from retrieved documentation. "
            "Cite evidence with bracketed markers like [1], [2] that correspond to "
            "the numbered snippets in the context. If the context is insufficient, "
            "say so explicitly. Be concise and technically precise."
        )
        user = (
            f"Context:\n{context_str if context_str else '(no documents retrieved)'}\n\n"
            f"Question: {query}\n\n"
            "Answer with inline citations like [1], [2]:"
        )

        answer = await self.provider.complete(
            prompt=user,
            system=system,
            temperature=0.3,
            max_tokens=1024,
        )
        return answer, sources

    async def _verify(self, query: str, answer: str) -> Dict[str, Any]:
        system = (
            "You are a verifier. Given a question and an answer, decide whether the "
            "answer directly addresses the question. Return ONLY JSON: "
            "{\"addresses\": true|false, \"reason\": \"short explanation\"}."
        )
        user = f'Question: "{query}"\n\nAnswer:\n{answer}\n\nReturn JSON only.'
        try:
            raw = await self.provider.complete(
                prompt=user, system=system, temperature=0.0, max_tokens=200
            )
            data = _extract_json(raw)
            if isinstance(data, dict) and "addresses" in data:
                return {
                    "addresses": bool(data.get("addresses")),
                    "reason": str(data.get("reason", "")),
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent._verify fallback (%s)", exc)
        return {"addresses": True, "reason": "verifier unavailable; assumed ok"}


# Lazy singleton
_agentic_rag: Optional[AgenticRAG] = None


def get_agentic_rag() -> AgenticRAG:
    global _agentic_rag
    if _agentic_rag is None:
        _agentic_rag = AgenticRAG()
    return _agentic_rag
