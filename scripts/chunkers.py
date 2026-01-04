#!/usr/bin/env python3
"""
Semantic Chunking Module for DevOps Documentation

Provides markdown-aware chunking that preserves:
- Heading hierarchy (h1, h2, h3, h4)
- Code blocks (never split mid-block)
- Structural context via metadata enrichment
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple

# Handle different LangChain import paths
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema.document import Document
    except ImportError:
        from langchain.schema import Document

try:
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
except ImportError:
    from langchain.text_splitter import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )


class ContentType(Enum):
    """Types of content detected in markdown documents"""
    PROSE = "prose"
    CODE = "code"
    LIST = "list"
    TABLE = "table"
    MIXED = "mixed"


@dataclass
class ChunkConfig:
    """Configuration for different content types"""
    # Default chunk sizes
    prose_chunk_size: int = 1000
    prose_chunk_overlap: int = 200

    # Code blocks typically need larger chunks to preserve context
    code_chunk_size: int = 1500
    code_chunk_overlap: int = 100

    # Lists work better with smaller chunks
    list_chunk_size: int = 800
    list_chunk_overlap: int = 150

    # Tables should rarely be split
    table_chunk_size: int = 2000
    table_chunk_overlap: int = 50

    # Minimum chunk size (don't create tiny chunks)
    min_chunk_size: int = 100

    # Headers to split on (markdown syntax -> metadata key)
    headers_to_split_on: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ])


class MarkdownSemanticChunker:
    """
    Markdown-aware semantic chunker that preserves document structure.

    Features:
    - Splits on markdown headers while preserving hierarchy
    - Detects and preserves code blocks
    - Enriches metadata with heading paths and content types
    - Configurable chunk sizes per content type

    Usage:
        chunker = MarkdownSemanticChunker()
        chunks = chunker.chunk_document(document)
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize the semantic chunker.

        Args:
            config: ChunkConfig instance with chunking parameters.
                   Uses defaults if not provided.
        """
        self.config = config or ChunkConfig()

        # Initialize the markdown header splitter
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.config.headers_to_split_on,
            strip_headers=False,  # Keep headers in content for context
        )

        # Initialize content-specific splitters
        self._init_content_splitters()

    def _init_content_splitters(self):
        """Initialize RecursiveCharacterTextSplitter for each content type"""
        self.splitters = {
            ContentType.PROSE: RecursiveCharacterTextSplitter(
                chunk_size=self.config.prose_chunk_size,
                chunk_overlap=self.config.prose_chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            ),
            ContentType.CODE: RecursiveCharacterTextSplitter(
                chunk_size=self.config.code_chunk_size,
                chunk_overlap=self.config.code_chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            ),
            ContentType.LIST: RecursiveCharacterTextSplitter(
                chunk_size=self.config.list_chunk_size,
                chunk_overlap=self.config.list_chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n- ", "\n* ", "\n", " ", ""],
            ),
            ContentType.TABLE: RecursiveCharacterTextSplitter(
                chunk_size=self.config.table_chunk_size,
                chunk_overlap=self.config.table_chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ""],
            ),
            ContentType.MIXED: RecursiveCharacterTextSplitter(
                chunk_size=self.config.prose_chunk_size,
                chunk_overlap=self.config.prose_chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            ),
        }

    def detect_content_type(self, text: str) -> ContentType:
        """
        Detect the primary content type of a text chunk.

        Args:
            text: The text content to analyze

        Returns:
            ContentType enum indicating the dominant content type
        """
        if not text.strip():
            return ContentType.PROSE

        lines = text.strip().split('\n')
        total_lines = len(lines)

        if total_lines == 0:
            return ContentType.PROSE

        # Count different content patterns
        code_block_pattern = re.compile(r'^```')
        list_pattern = re.compile(r'^[\s]*[-*+]\s|^[\s]*\d+\.\s')
        table_pattern = re.compile(r'^\|.*\|')

        code_lines = 0
        list_lines = 0
        table_lines = 0
        in_code_block = False

        for line in lines:
            if code_block_pattern.match(line):
                in_code_block = not in_code_block
                code_lines += 1
            elif in_code_block:
                code_lines += 1
            elif list_pattern.match(line):
                list_lines += 1
            elif table_pattern.match(line):
                table_lines += 1

        # Determine dominant type (>50% of content)
        threshold = total_lines * 0.5

        if code_lines > threshold:
            return ContentType.CODE
        elif table_lines > threshold:
            return ContentType.TABLE
        elif list_lines > threshold:
            return ContentType.LIST
        elif code_lines > 0 or table_lines > 0 or list_lines > 0:
            return ContentType.MIXED
        else:
            return ContentType.PROSE

    def build_heading_path(self, metadata: Dict[str, str]) -> str:
        """
        Build a hierarchical heading path from metadata.

        Args:
            metadata: Dictionary containing h1, h2, h3, h4 keys

        Returns:
            String like "Kubernetes > Pods > Lifecycle"
        """
        path_parts = []
        for level in ['h1', 'h2', 'h3', 'h4']:
            if level in metadata and metadata[level]:
                # Clean up the heading (remove markdown syntax if present)
                heading = metadata[level].strip()
                heading = re.sub(r'^#+\s*', '', heading)  # Remove leading #
                if heading:
                    path_parts.append(heading)

        return " > ".join(path_parts) if path_parts else ""

    def extract_code_blocks(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Extract code blocks from text with their positions.

        Args:
            text: The markdown text

        Returns:
            List of tuples: (code_content, language, start_pos, end_pos)
        """
        code_blocks = []
        # Match fenced code blocks with optional language
        pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)

        for match in pattern.finditer(text):
            language = match.group(1) or "unknown"
            code_content = match.group(2)
            code_blocks.append((
                code_content,
                language,
                match.start(),
                match.end()
            ))

        return code_blocks

    def preserve_code_blocks(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replace code blocks with placeholders to prevent splitting.

        Args:
            text: Original text with code blocks

        Returns:
            Tuple of (text_with_placeholders, placeholder_map)
        """
        placeholder_map = {}
        code_blocks = self.extract_code_blocks(text)

        # Process in reverse order to maintain positions
        modified_text = text
        for i, (content, language, start, end) in enumerate(reversed(code_blocks)):
            placeholder = f"__CODE_BLOCK_{len(code_blocks) - 1 - i}__"
            # Keep the full code block as the replacement value
            original_block = text[start:end]
            placeholder_map[placeholder] = original_block
            modified_text = modified_text[:start] + placeholder + modified_text[end:]

        return modified_text, placeholder_map

    def restore_code_blocks(self, text: str, placeholder_map: Dict[str, str]) -> str:
        """
        Restore code blocks from placeholders.

        Args:
            text: Text with placeholders
            placeholder_map: Map of placeholder -> original code block

        Returns:
            Text with code blocks restored
        """
        result = text
        for placeholder, code_block in placeholder_map.items():
            result = result.replace(placeholder, code_block)
        return result

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a single document using semantic markdown splitting.

        Args:
            document: LangChain Document object

        Returns:
            List of Document objects with enriched metadata
        """
        text = document.page_content
        original_metadata = document.metadata.copy()

        # Step 1: Preserve code blocks
        text_with_placeholders, code_map = self.preserve_code_blocks(text)

        # Step 2: Split by markdown headers
        try:
            header_splits = self.header_splitter.split_text(text_with_placeholders)
        except Exception as e:
            # Fallback: treat entire document as one section
            print(f"Warning: Header splitting failed, using fallback: {e}")
            header_splits = [Document(
                page_content=text_with_placeholders,
                metadata={}
            )]

        # Step 3: Process each header-based section
        final_chunks = []

        for section in header_splits:
            section_text = section.page_content
            section_metadata = section.metadata.copy()

            # Restore code blocks in this section
            section_text = self.restore_code_blocks(section_text, code_map)

            # Skip empty sections
            if not section_text.strip():
                continue

            # Detect content type
            content_type = self.detect_content_type(section_text)

            # Build heading path
            heading_path = self.build_heading_path(section_metadata)

            # Get appropriate splitter
            splitter = self.splitters[content_type]

            # Check if we need to split further
            if len(section_text) <= self.config.min_chunk_size:
                # Section is small enough, keep as single chunk
                sub_chunks = [section_text]
            elif len(section_text) <= splitter._chunk_size:
                # Section fits in one chunk
                sub_chunks = [section_text]
            else:
                # Need to split the section
                # For code-heavy sections, try to split at code block boundaries
                if content_type == ContentType.CODE:
                    sub_chunks = self._split_preserving_code(section_text, splitter)
                else:
                    sub_chunks = splitter.split_text(section_text)

            # Create Document objects with enriched metadata
            for i, chunk_text in enumerate(sub_chunks):
                if not chunk_text.strip():
                    continue

                # Merge original metadata with section metadata
                chunk_metadata = {
                    **original_metadata,
                    **section_metadata,
                    'heading_path': heading_path,
                    'content_type': content_type.value,
                    'chunk_index': i,
                    'total_chunks_in_section': len(sub_chunks),
                }

                # Add code language if present
                if content_type == ContentType.CODE:
                    code_blocks = self.extract_code_blocks(chunk_text)
                    if code_blocks:
                        languages = list(set(cb[1] for cb in code_blocks if cb[1] != "unknown"))
                        if languages:
                            chunk_metadata['code_languages'] = ",".join(languages)

                final_chunks.append(Document(
                    page_content=chunk_text.strip(),
                    metadata=chunk_metadata
                ))

        # Fallback if no chunks were created
        if not final_chunks:
            final_chunks.append(Document(
                page_content=text.strip() if text.strip() else document.page_content,
                metadata={
                    **original_metadata,
                    'heading_path': '',
                    'content_type': ContentType.PROSE.value,
                    'chunk_index': 0,
                    'total_chunks_in_section': 1,
                }
            ))

        return final_chunks

    def _split_preserving_code(self, text: str, splitter: RecursiveCharacterTextSplitter) -> List[str]:
        """
        Split text while trying to keep code blocks intact.

        Args:
            text: Text to split
            splitter: The splitter to use

        Returns:
            List of text chunks
        """
        # Find code block boundaries
        code_blocks = self.extract_code_blocks(text)

        if not code_blocks:
            return splitter.split_text(text)

        chunks = []
        current_pos = 0

        for content, language, start, end in code_blocks:
            # Process text before code block
            if start > current_pos:
                before_text = text[current_pos:start].strip()
                if before_text:
                    if len(before_text) > splitter._chunk_size:
                        chunks.extend(splitter.split_text(before_text))
                    else:
                        chunks.append(before_text)

            # Add code block as its own chunk (don't split it)
            code_block_text = text[start:end]
            if len(code_block_text) > self.config.code_chunk_size * 2:
                # Code block is very large, we may need to split it
                # Split at function/class boundaries if possible
                chunks.extend(self._split_large_code_block(code_block_text))
            else:
                chunks.append(code_block_text)

            current_pos = end

        # Process remaining text after last code block
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                if len(remaining) > splitter._chunk_size:
                    chunks.extend(splitter.split_text(remaining))
                else:
                    chunks.append(remaining)

        return chunks

    def _split_large_code_block(self, code_block: str) -> List[str]:
        """
        Split a very large code block at logical boundaries.

        Args:
            code_block: The code block text (including ``` markers)

        Returns:
            List of code chunks
        """
        # Extract language and content
        match = re.match(r'```(\w*)\n(.*?)```', code_block, re.DOTALL)
        if not match:
            return [code_block]

        language = match.group(1)
        content = match.group(2)

        # Try to split at logical boundaries
        # Common patterns: empty lines, function definitions, class definitions
        split_patterns = [
            r'\n\n+',  # Double newlines
            r'\n(?=def\s)',  # Python function
            r'\n(?=class\s)',  # Python class
            r'\n(?=func\s)',  # Go function
            r'\n(?=function\s)',  # JavaScript function
            r'\n(?=resource\s)',  # Terraform resource
            r'\n(?=apiVersion:)',  # Kubernetes manifest
        ]

        chunks = []
        lines = content.split('\n')
        current_chunk_lines = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > self.config.code_chunk_size and current_chunk_lines:
                # Save current chunk
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append(f"```{language}\n{chunk_content}\n```")
                current_chunk_lines = [line]
                current_size = line_size
            else:
                current_chunk_lines.append(line)
                current_size += line_size

        # Don't forget the last chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunks.append(f"```{language}\n{chunk_content}\n```")

        return chunks if chunks else [code_block]

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk multiple documents.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of chunked Document objects with enriched metadata
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks


def create_chunker_from_env() -> MarkdownSemanticChunker:
    """
    Create a MarkdownSemanticChunker with configuration from environment variables.

    Environment variables:
        CHUNK_SIZE: Base chunk size (default 1000)
        CHUNK_OVERLAP: Base chunk overlap (default 200)
        CODE_CHUNK_SIZE: Chunk size for code blocks (default 1500)
        CODE_CHUNK_OVERLAP: Overlap for code blocks (default 100)
        LIST_CHUNK_SIZE: Chunk size for lists (default 800)
        TABLE_CHUNK_SIZE: Chunk size for tables (default 2000)
        MIN_CHUNK_SIZE: Minimum chunk size (default 100)

    Returns:
        Configured MarkdownSemanticChunker instance
    """
    import os

    config = ChunkConfig(
        prose_chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        prose_chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        code_chunk_size=int(os.getenv("CODE_CHUNK_SIZE", 1500)),
        code_chunk_overlap=int(os.getenv("CODE_CHUNK_OVERLAP", 100)),
        list_chunk_size=int(os.getenv("LIST_CHUNK_SIZE", 800)),
        list_chunk_overlap=int(os.getenv("LIST_CHUNK_OVERLAP", 150)),
        table_chunk_size=int(os.getenv("TABLE_CHUNK_SIZE", 2000)),
        table_chunk_overlap=int(os.getenv("TABLE_CHUNK_OVERLAP", 50)),
        min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", 100)),
    )

    return MarkdownSemanticChunker(config)


# Convenience function for backward compatibility
def get_semantic_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> MarkdownSemanticChunker:
    """
    Get a semantic chunker with custom base settings.

    Args:
        chunk_size: Base chunk size for prose content
        chunk_overlap: Base overlap for prose content

    Returns:
        Configured MarkdownSemanticChunker
    """
    config = ChunkConfig(
        prose_chunk_size=chunk_size,
        prose_chunk_overlap=chunk_overlap,
        # Scale other sizes proportionally
        code_chunk_size=int(chunk_size * 1.5),
        list_chunk_size=int(chunk_size * 0.8),
        table_chunk_size=int(chunk_size * 2),
    )

    return MarkdownSemanticChunker(config)
