"""Contextual compression for retrieved documents.

Extracts only query-relevant passages from retrieved chunks to:
- Reduce context window usage by 40-60%
- Improve response quality by removing irrelevant content
- Better utilize limited local LLM context windows
"""

import logging
import re
from typing import List, Optional
from dataclasses import dataclass
import ollama

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of context compression."""
    original_length: int
    compressed_length: int
    compression_ratio: float
    compressed_content: str


class ContextCompressor:
    """Compresses retrieved chunks to extract query-relevant passages."""

    # Prompt for LLM-based compression
    COMPRESSION_PROMPT = '''Extract only the sentences from the following document that directly help answer this question. Return the relevant excerpts verbatim, preserving code blocks. If nothing is relevant, return "NONE".

Question: {query}

Document:
{document}

Relevant excerpts:'''

    # Simple extractive compression using keyword matching
    def compress_simple(self, query: str, content: str, max_sentences: int = 10) -> CompressionResult:
        """Fast keyword-based extraction without LLM call."""
        original_length = len(content)

        # Extract query keywords (remove common words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'how', 'what', 'why', 'do', 'does', 'to', 'in', 'for', 'of', 'and', 'or'}
        query_words = set(w.lower() for w in re.findall(r'\w+', query) if w.lower() not in stop_words and len(w) > 2)

        # Split into sentences (preserve code blocks)
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        text_without_code = re.sub(r'```[\s\S]*?```', '{{CODE_BLOCK}}', content)

        sentences = re.split(r'(?<=[.!?])\s+', text_without_code)

        # Score sentences by keyword overlap
        scored = []
        for sent in sentences:
            if '{{CODE_BLOCK}}' in sent:
                scored.append((sent, 10))  # Keep code blocks
            else:
                sent_words = set(w.lower() for w in re.findall(r'\w+', sent))
                score = len(query_words & sent_words)
                if score > 0:
                    scored.append((sent, score))

        # Sort by score and take top sentences
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [s[0] for s in scored[:max_sentences]]

        # Restore code blocks
        compressed = ' '.join(selected)
        for code_block in code_blocks:
            compressed = compressed.replace('{{CODE_BLOCK}}', code_block, 1)

        compressed_length = len(compressed)

        return CompressionResult(
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=compressed_length / original_length if original_length > 0 else 1.0,
            compressed_content=compressed if compressed.strip() else content
        )

    def compress_with_llm(self, query: str, content: str, model: str = None) -> CompressionResult:
        """LLM-based compression for higher quality extraction."""
        original_length = len(content)
        model = model or settings.ollama_default_model

        # Truncate very long content to avoid overwhelming the model
        if len(content) > 3000:
            content = content[:3000] + "..."

        try:
            prompt = self.COMPRESSION_PROMPT.format(query=query, document=content)
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 500, "temperature": 0.1}
            )

            compressed = response["message"]["content"].strip()

            # If model returns NONE or empty, fall back to simple compression
            if not compressed or compressed.upper() == "NONE":
                return self.compress_simple(query, content)

            compressed_length = len(compressed)

            return CompressionResult(
                original_length=original_length,
                compressed_length=compressed_length,
                compression_ratio=compressed_length / original_length if original_length > 0 else 1.0,
                compressed_content=compressed
            )

        except Exception as e:
            logger.warning(f"LLM compression failed, falling back to simple: {e}")
            return self.compress_simple(query, content)

    def compress_chunks(
        self,
        query: str,
        chunks: List[str],
        use_llm: bool = False,
        max_chunks: int = 5
    ) -> List[CompressionResult]:
        """Compress multiple chunks."""
        results = []
        compress_fn = self.compress_with_llm if use_llm else self.compress_simple

        for chunk in chunks[:max_chunks]:
            result = compress_fn(query, chunk)
            results.append(result)

        return results


# Singleton instance
context_compressor = ContextCompressor()
