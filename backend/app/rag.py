"""RAG (Retrieval-Augmented Generation) pipeline with reranking, score-aware retrieval, and metrics.

This module implements a production-ready RAG pipeline with:
- Score-aware vector retrieval with configurable thresholds
- Optional cross-encoder reranking for improved relevance
- Comprehensive metrics logging for observability
- Streaming response support
- Circuit breaker protection for external service calls

Flow:
    Query -> Vector Search (top 20) -> Rerank (top 5) -> Build Context -> LLM
"""
import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
import hashlib
from typing import List, Optional, Dict, Any, Tuple
import ollama

from app.config import settings
from app.vectorstore import vector_store
from app.reranker import rerank_documents, get_reranker
from app.metrics import retrieval_metrics_logger, RetrievalTimer, RetrievalMetrics
from app.query_expansion import hyde_expander
from app.web_search import web_searcher
from app.conversation_context import conversation_expander
from app.circuit_breaker import (
    ollama_circuit_breaker,
    CircuitBreakerOpen,
    is_circuit_breaker_exception,
    get_service_unavailable_message,
)
from app.output_validation import validate_response, ValidationResult
from app.tracing import get_tracer, create_span, SpanAttributes
from app.drift_detection import drift_detector

logger = logging.getLogger(__name__)


# Model-specific context window limits (tokens)
# Using ~75% of actual limit to leave room for response generation
MODEL_CONTEXT_LIMITS = {
    "llama3.1": 96000,      # 128K context * 0.75
    "llama3.2": 96000,      # 128K context * 0.75
    "llama3": 6144,         # 8K context * 0.75
    "mistral": 24576,       # 32K context * 0.75
    "mixtral": 24576,       # 32K context * 0.75
    "qwen2.5": 24576,       # 32K context * 0.75
    "qwen2.5-coder": 24576, # 32K context * 0.75
    "codellama": 12288,     # 16K context * 0.75
    "deepseek": 24576,      # 32K context * 0.75
    "phi": 3072,            # 4K context * 0.75
}

# Default context token budget (fallback for unknown models)
DEFAULT_MAX_CONTEXT_TOKENS = 4096


def get_model_context_limit(model_name: str) -> int:
    """Get context token limit based on model capabilities.

    Returns ~75% of model's context window to leave room for response generation.

    Args:
        model_name: The model name/identifier (e.g., "llama3.1:8b", "mistral:7b")

    Returns:
        Token limit for context window. Falls back to DEFAULT_MAX_CONTEXT_TOKENS
        if model is not recognized.
    """
    if not model_name:
        return DEFAULT_MAX_CONTEXT_TOKENS

    model_lower = model_name.lower()
    # Check patterns in order (more specific patterns first due to dict ordering)
    for pattern, limit in MODEL_CONTEXT_LIMITS.items():
        if pattern in model_lower:
            return limit
    return DEFAULT_MAX_CONTEXT_TOKENS  # fallback


# Few-shot examples for domain-specific response formatting
# Each example demonstrates the expected output format with code blocks, source citations, and structure
FEW_SHOT_EXAMPLES = {
    "kubernetes": {
        "query": "How do I check pod logs?",
        "context": "kubectl logs retrieves container logs. Use -f for streaming, --previous for crashed containers.",
        "response": """To check pod logs in Kubernetes:

```bash
# Basic log retrieval
kubectl logs <pod-name>

# Stream logs in real-time
kubectl logs -f <pod-name>

# View logs from previous container instance (useful for crashed pods)
kubectl logs --previous <pod-name>

# For multi-container pods, specify the container
kubectl logs <pod-name> -c <container-name>
```

[Source 1] The `-f` flag follows the log output similar to `tail -f`."""
    },
    "docker": {
        "query": "How do I build a Docker image?",
        "context": "docker build creates images from Dockerfiles. Use -t for tagging.",
        "response": """To build a Docker image:

```bash
# Build from current directory
docker build -t myimage:latest .

# Build with specific Dockerfile
docker build -f Dockerfile.prod -t myimage:prod .

# Build with build arguments
docker build --build-arg VERSION=1.0 -t myimage:1.0 .
```

[Source 1] Always tag images with meaningful versions for traceability."""
    },
    "terraform": {
        "query": "How do I initialize Terraform?",
        "context": "terraform init initializes working directory, downloads providers.",
        "response": """To initialize a Terraform project:

```bash
# Initialize the working directory
terraform init

# Upgrade providers to latest versions
terraform init -upgrade

# Reconfigure backend
terraform init -reconfigure
```

[Source 1] Run `terraform init` whenever you add new providers or modules."""
    },
    "ansible": {
        "query": "How do I run an Ansible playbook?",
        "context": "ansible-playbook runs playbooks against inventory hosts. Use -i for inventory, --check for dry run.",
        "response": """To run an Ansible playbook:

```bash
# Run playbook with default inventory
ansible-playbook playbook.yml

# Specify custom inventory file
ansible-playbook -i inventory.ini playbook.yml

# Dry run (check mode) - shows changes without applying
ansible-playbook playbook.yml --check

# Run with extra variables
ansible-playbook playbook.yml -e "env=production"
```

[Source 1] Use `--check` mode to preview changes before applying them."""
    },
    "helm": {
        "query": "How do I install a Helm chart?",
        "context": "helm install deploys charts to Kubernetes. Use -f for custom values, --namespace for target namespace.",
        "response": """To install a Helm chart:

```bash
# Install chart with release name
helm install my-release bitnami/nginx

# Install with custom values file
helm install my-release bitnami/nginx -f values.yaml

# Install in specific namespace
helm install my-release bitnami/nginx --namespace production --create-namespace

# Install with inline value overrides
helm install my-release bitnami/nginx --set replicaCount=3
```

[Source 1] Use `-f values.yaml` to customize chart configuration."""
    },
    "cicd": {
        "query": "How do I set up a GitHub Actions workflow?",
        "context": "GitHub Actions workflows are defined in .github/workflows/*.yml. Jobs run in parallel by default.",
        "response": """To set up a GitHub Actions workflow:

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: |
          npm install
          npm test
```

[Source 1] Workflows trigger on events like `push` and `pull_request`."""
    },
}

# Keyword patterns for detecting domain from query
# Maps keyword patterns to few-shot example keys
FEW_SHOT_DOMAIN_PATTERNS = [
    # Kubernetes
    (["kubernetes", "k8s", "kubectl", "pod", "deployment", "service", "ingress", "namespace", "configmap", "secret", "pvc", "pv", "statefulset", "daemonset", "replicaset"], "kubernetes"),
    # Docker
    (["docker", "dockerfile", "container", "image", "docker-compose", "docker compose"], "docker"),
    # Terraform
    (["terraform", "tf", "hcl", "tfstate", "tfvars", "provider", "resource", "module"], "terraform"),
    # Ansible
    (["ansible", "playbook", "ansible-playbook", "inventory", "role", "task", "handler", "ansible-vault"], "ansible"),
    # Helm
    (["helm", "chart", "values.yaml", "helmfile", "helm install", "helm upgrade"], "helm"),
    # CI/CD
    (["github actions", "gitlab ci", "jenkins", "pipeline", "workflow", "ci/cd", "cicd", "build pipeline", "deploy pipeline"], "cicd"),
]


def select_few_shot_example(query: str) -> Optional[Dict[str, str]]:
    """Select a relevant few-shot example based on query content.

    Analyzes the query for domain-specific keywords and returns
    a matching example if found. Used to improve output consistency
    by providing the LLM with a formatting reference.

    Args:
        query: The user's question/query string

    Returns:
        Dictionary with 'query', 'context', 'response' keys if a relevant
        example is found, None otherwise.
    """
    query_lower = query.lower()

    for keywords, domain in FEW_SHOT_DOMAIN_PATTERNS:
        for keyword in keywords:
            if keyword in query_lower:
                example = FEW_SHOT_EXAMPLES.get(domain)
                if example:
                    if settings.log_retrieval_details:
                        logger.info(f"Selected few-shot example for domain: {domain} (matched: '{keyword}')")
                    return example

    return None


@dataclass
class RetrievalResult:
    """Container for retrieval results with scores and performance metrics."""
    documents: List[Any] = field(default_factory=list)
    similarity_scores: List[float] = field(default_factory=list)
    rerank_scores: List[float] = field(default_factory=list)
    initial_count: int = 0
    final_count: int = 0
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0
    reranker_used: bool = False
    reranker_model: Optional[str] = None
    retrieval_error: Optional[str] = None
    rerank_error: Optional[str] = None
    # Hybrid search fields
    hybrid_search_used: bool = False
    dense_count: int = 0
    sparse_count: int = 0
    # HyDE (Hypothetical Document Embeddings) fields
    hyde_used: bool = False
    hyde_time_ms: float = 0.0
    hyde_skipped_reason: Optional[str] = None
    # Web search fallback fields
    web_search_used: bool = False
    web_search_time_ms: float = 0.0
    web_search_results_count: int = 0
    web_search_trigger_reason: Optional[str] = None
    web_search_error: Optional[str] = None
    web_search_context: str = ""  # Formatted web results for context
    # Embedding cache tracking
    embedding_cache_hit: Optional[bool] = None
    # Conversation context fields
    conversation_context_used: bool = False
    conversation_context_terms: List[str] = field(default_factory=list)
    original_query: Optional[str] = None  # Query before context expansion


class RAGPipeline:
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.default_model = settings.ollama_default_model

    def _count_tokens(self, text: str) -> int:
        """Estimate token count for a text string.

        Uses a rough approximation of ~4 characters per token, which is
        a reasonable estimate for English text with code snippets.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def _format_context(
        self,
        documents: List,
        web_context: str = "",
        max_context_tokens: Optional[int] = None
    ) -> str:
        """Format retrieved documents into context string with optional truncation.

        Args:
            documents: List of Document objects from vector store (ordered by relevance)
            web_context: Optional formatted web search results
            max_context_tokens: Maximum tokens for context. If None, uses DEFAULT_MAX_CONTEXT_TOKENS.
                               Documents are added in order of relevance until budget is exhausted.

        Returns:
            Combined context string for LLM, truncated to fit within token budget
        """
        if max_context_tokens is None:
            max_context_tokens = DEFAULT_MAX_CONTEXT_TOKENS

        context_parts = []
        current_tokens = 0
        separator = "\n---\n"
        separator_tokens = self._count_tokens(separator)

        # Reserve tokens for web context if present
        web_context_tokens = 0
        web_prefix = "\n\n--- Web Search Results ---\n\n"
        if web_context:
            web_context_tokens = self._count_tokens(web_prefix + web_context)

        available_tokens = max_context_tokens - web_context_tokens

        # Add local document context in order of relevance (higher-ranked first)
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            source_type = doc.metadata.get('source_type', 'Unknown')
            content = doc.page_content.strip()

            doc_text = f"[Source {i} - {source_type}]\n{content}\n"
            doc_tokens = self._count_tokens(doc_text)

            # Account for separator between documents
            needed_tokens = doc_tokens
            if context_parts:
                needed_tokens += separator_tokens

            # Check if adding this document would exceed the budget
            if current_tokens + needed_tokens > available_tokens:
                # Try to add a truncated version if we have room for at least some content
                remaining_tokens = available_tokens - current_tokens
                if context_parts:
                    remaining_tokens -= separator_tokens

                # Only truncate if we can include meaningful content (at least 50 tokens)
                if remaining_tokens >= 50:
                    # Estimate characters from tokens (reverse of _count_tokens)
                    remaining_chars = remaining_tokens * 4
                    header = f"[Source {i} - {source_type}]\n"
                    content_budget = remaining_chars - len(header) - 20  # -20 for safety margin

                    if content_budget > 100:
                        truncated_content = content[:content_budget] + "..."
                        doc_text = f"{header}{truncated_content}\n"
                        context_parts.append(doc_text)
                        logger.debug(
                            f"Context truncation: Document {i} truncated to {content_budget} chars "
                            f"(budget: {max_context_tokens} tokens)"
                        )
                break

            context_parts.append(doc_text)
            current_tokens += needed_tokens

        local_context = separator.join(context_parts) if context_parts else ""

        # Add web search context if available
        if web_context:
            if local_context:
                return f"{local_context}{web_prefix}{web_context}"
            else:
                return f"--- Web Search Results ---\n\n{web_context}"

        return local_context
    
    # Model-specific system prompts optimized for different LLM families
    # Each prompt is tailored to the model's instruction-following style and known strengths
    MODEL_SPECIFIC_PROMPTS = {
        # Llama 3.x family - Strong at structured output, follows instructions well
        # Optimized for their training format and reasoning capabilities
        "llama3": """You are an expert DevOps engineer assistant specializing in infrastructure, cloud, and automation.

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

ROLE: Expert DevOps engineer providing accurate, actionable technical guidance.

CORE BEHAVIOR:
1. Answer using provided context as primary source
2. Clearly state when using general knowledge beyond context
3. Provide working code examples with proper syntax
4. Be direct and technically precise

OUTPUT REQUIREMENTS:
- Use markdown: headers (##), code blocks (```language), bullet lists
- Cite sources as [Source N] when referencing context
- Code blocks MUST include language identifier
- Prefer practical examples over theoretical explanations

BOUNDARIES:
- DevOps, infrastructure, cloud, CI/CD, containers, orchestration, monitoring topics only
- Decline off-topic requests politely, redirect to DevOps questions
- Never assist with unauthorized access or security exploitation
- When uncertain, acknowledge limitations""",

        # Mistral/Mixtral family - Efficient, good at concise responses
        # Optimized for their instruction format and speed
        "mistral": """[INST] You are an expert DevOps engineer assistant. Provide concise, accurate technical guidance.

Guidelines:
- Primary source: provided context documents
- State clearly when drawing from general knowledge
- Include working code examples with syntax highlighting
- Be efficient - value clarity over verbosity

Format requirements:
- Markdown formatting (headers, lists, code blocks)
- Source citations: [Source N]
- All code in fenced blocks with language tags
- Structure complex answers with clear sections

Scope: DevOps, infrastructure, cloud, CI/CD, containers, monitoring, automation
Decline: Off-topic requests, security exploits, unauthorized access guidance
Uncertainty: Acknowledge when unsure [/INST]""",

        # Qwen family - Strong multilingual, good at detailed explanations
        # Optimized for their training style emphasizing helpful, detailed responses
        "qwen": """<|im_start|>system
You are an expert DevOps engineer assistant with deep knowledge of infrastructure, cloud platforms, and automation tools.

### Your Responsibilities:
1. **Context-First**: Base answers on provided documentation context
2. **Transparency**: Explicitly note when using knowledge beyond provided context
3. **Practical Focus**: Provide executable code examples and step-by-step guidance
4. **Precision**: Use accurate technical terminology and current best practices

### Response Format:
- Structure with markdown headers for complex topics
- All code in fenced blocks with language specifiers (```yaml, ```bash, etc.)
- Reference sources as [Source N] when citing provided context
- Use bullet points for lists, numbered steps for procedures

### Topic Scope:
ALLOWED: DevOps, cloud infrastructure, CI/CD pipelines, containerization, Kubernetes, monitoring, logging, automation, IaC
NOT ALLOWED: Non-technical topics, security exploitation, unauthorized access

### When Uncertain:
State limitations clearly and suggest alternative resources if appropriate.
<|im_end|>""",

        # Default fallback for unknown models - Generic but effective
        "default": """You are an expert DevOps engineer assistant. Your role is to help users with DevOps, infrastructure, and programming questions.

## INSTRUCTIONS

- Answer based primarily on the provided context when available
- If the context doesn't fully answer the question, use your general knowledge but mention this
- Provide code examples when relevant, using proper markdown code blocks
- Be concise but thorough
- If you're unsure, say so

## OUTPUT FORMAT

- Use markdown formatting for readability (headers, lists, code blocks)
- Include source citations as [Source N] when referencing provided context
- Keep responses concise and focused - avoid unnecessary verbosity
- Structure longer responses with clear sections using markdown headers
- Use code blocks with language specifiers for all code examples (e.g., ```yaml, ```bash)

## SAFETY BOUNDARIES

- Only answer questions related to DevOps, infrastructure, cloud computing, CI/CD, containerization, orchestration, monitoring, and related technical topics
- For off-topic requests (personal advice, creative writing, general trivia, etc.), politely decline and redirect the user to ask DevOps-related questions
- Do not provide assistance with malicious activities such as unauthorized access, exploiting vulnerabilities, or bypassing security controls
- If a question is ambiguous, interpret it in the context of DevOps best practices"""
    }

    # Model family detection patterns - maps regex patterns to prompt keys
    MODEL_FAMILY_PATTERNS = [
        # Llama 3.x family (llama3, llama3.1, llama3.2, llama3.3, etc.)
        (r"llama-?3", "llama3"),
        # Mistral family (mistral, mixtral, codestral)
        (r"mistral|mixtral|codestral", "mistral"),
        # Qwen family (qwen, qwen2, qwen2.5, qwen-coder, etc.)
        (r"qwen", "qwen"),
    ]

    def _detect_model_family(self, model_name: str) -> str:
        """Detect the model family from the model name.

        Uses regex patterns to match model names to their families.
        Returns the prompt key to use for the detected family.

        Args:
            model_name: The model name/identifier (e.g., "llama3.1:8b", "mistral:7b")

        Returns:
            The prompt key for the detected family, or "default" if unknown
        """
        model_lower = model_name.lower()

        for pattern, family_key in self.MODEL_FAMILY_PATTERNS:
            if re.search(pattern, model_lower):
                return family_key

        return "default"

    def _get_system_prompt(self, model: Optional[str] = None) -> str:
        """Get the system prompt optimized for the specified model family.

        Auto-detects model family from the model name and returns an
        optimized system prompt tailored to that family's instruction
        following style and known strengths.

        Args:
            model: The model name (e.g., "llama3.1:8b"). If None, uses default model.

        Returns:
            System prompt string optimized for the model family
        """
        model = model or self.default_model
        family = self._detect_model_family(model)
        prompt = self.MODEL_SPECIFIC_PROMPTS.get(family, self.MODEL_SPECIFIC_PROMPTS["default"])

        # Log the model family detection for observability
        if settings.log_retrieval_details:
            if family != "default":
                logger.info(f"Using {family}-optimized system prompt for model: {model}")
            else:
                logger.info(f"Using default system prompt for unrecognized model: {model}")

        return prompt

    def _get_user_prompt(self, query: str, context: str) -> str:
        """Build the user prompt with context and query."""
        if context:
            return f"""Context from documentation:
{context}

Question: {query}"""
        else:
            return query

    def _build_messages(self, query: str, context: str, model: Optional[str] = None) -> List[Dict[str, str]]:
        """Build messages list with proper system/user role separation.

        Optionally includes few-shot examples as user/assistant message pairs
        to improve output consistency and formatting. Examples are selected
        based on domain-specific keywords in the query.

        Args:
            query: The user's question
            context: Formatted context from retrieved documents
            model: The model name for selecting the appropriate system prompt

        Returns:
            List of message dictionaries with system, user, and optional assistant roles
        """
        messages = [
            {'role': 'system', 'content': self._get_system_prompt(model)},
        ]

        # Add few-shot example if enabled and relevant example found
        if settings.few_shot_enabled:
            example = select_few_shot_example(query)
            if example:
                # Build example user prompt in same format as real queries
                example_user_prompt = self._get_user_prompt(
                    query=example['query'],
                    context=example['context']
                )
                messages.append({'role': 'user', 'content': example_user_prompt})
                messages.append({'role': 'assistant', 'content': example['response']})

        # Add actual user query
        messages.append({'role': 'user', 'content': self._get_user_prompt(query, context)})

        return messages
    
    def _retrieve_with_scores(
        self,
        query: str,
        model: str = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> RetrievalResult:
        """Retrieve documents with similarity scores and optional reranking.

        Implements the core retrieval flow:
        0. (Optional) Expand query with conversation context for follow-up questions
        1. Vector search for initial candidates (retrieval_top_k if reranking, else top_k)
        2. Apply minimum similarity score threshold
        3. Optional: Rerank candidates with cross-encoder
        4. Return final top_k results with all scores

        Args:
            query: The search query string
            model: The LLM model being used (for metrics logging)
            conversation_history: Optional list of prior messages for context-aware retrieval

        Returns:
            RetrievalResult with documents, scores, timing, and metadata
        """
        model = model or self.default_model
        result = RetrievalResult()
        total_start = time.perf_counter()
        tracer = get_tracer()

        # Compute query hash for tracing correlation
        query_hash = hashlib.md5(query.encode()).hexdigest()[:12]

        # Determine initial retrieval count based on whether reranking is enabled
        if settings.reranker_enabled:
            initial_top_k = settings.retrieval_top_k
        else:
            initial_top_k = settings.top_k_results

        # Phase -1: Conversation context expansion (if enabled and history provided)
        # This resolves pronouns and references like "it", "that", "the same thing"
        search_query = query
        if settings.conversation_context_enabled and conversation_history:
            context_result = conversation_expander.expand_query(query, conversation_history)
            if context_result.expanded:
                result.original_query = query
                search_query = context_result.expanded_query
                result.conversation_context_used = True
                result.conversation_context_terms = context_result.context_terms
                if settings.log_retrieval_details:
                    logger.info(
                        f"Conversation context expanded query: '{query}' -> '{search_query}' "
                        f"(terms: {context_result.context_terms})"
                    )

        # Phase 0: HyDE query expansion (if enabled)
        # Note: HyDE operates on the already context-expanded query
        if settings.hyde_enabled:
            with tracer.start_as_current_span("rag.hyde") as hyde_span:
                hyde_span.set_attribute(SpanAttributes.HYDE_ENABLED, True)
                hyde_span.set_attribute(SpanAttributes.QUERY_LENGTH, len(search_query))

                hyde_result = hyde_expander.expand_sync(search_query)
                result.hyde_time_ms = hyde_result.generation_time_ms

                if hyde_result.expanded and hyde_result.hypothetical_document:
                    # Use hypothetical document for retrieval (better semantic match)
                    search_query = hyde_result.hypothetical_document
                    result.hyde_used = True
                    hyde_span.set_attribute(SpanAttributes.HYDE_USED, True)
                    hyde_span.set_attribute(SpanAttributes.HYDE_TIME_MS, result.hyde_time_ms)
                    if settings.log_retrieval_details:
                        logger.info(
                            f"HyDE expanded query in {result.hyde_time_ms:.1f}ms: "
                            f"'{query[:50]}...' -> {len(hyde_result.hypothetical_document)} chars"
                        )
                else:
                    result.hyde_skipped_reason = hyde_result.skip_reason or hyde_result.error
                    hyde_span.set_attribute(SpanAttributes.HYDE_USED, False)
                    hyde_span.set_attribute(SpanAttributes.HYDE_SKIP_REASON, result.hyde_skipped_reason or "unknown")

        # Phase 1: Vector search with scores (hybrid or dense-only)
        with tracer.start_as_current_span("rag.retrieval") as retrieval_span:
            retrieval_span.set_attribute(SpanAttributes.QUERY_HASH, query_hash)
            retrieval_span.set_attribute(SpanAttributes.QUERY_LENGTH, len(query))
            retrieval_span.set_attribute(SpanAttributes.RETRIEVAL_TOP_K, initial_top_k)
            retrieval_span.set_attribute(SpanAttributes.RETRIEVAL_HYBRID, settings.hybrid_search_enabled)

            retrieval_start = time.perf_counter()
            try:
                # Use hybrid search if enabled, otherwise dense-only
                # Both methods now return (results, cache_hit) tuple
                if settings.hybrid_search_enabled:
                    results_with_scores, cache_hit = vector_store.hybrid_search_with_cache_info(
                        query=search_query,
                        top_k=initial_top_k,
                        min_score=settings.min_similarity_score
                    )
                    result.hybrid_search_used = True
                else:
                    results_with_scores, cache_hit = vector_store.search_with_cache_info(
                        query=search_query,
                        top_k=initial_top_k,
                        min_score=settings.min_similarity_score
                    )
                result.embedding_cache_hit = cache_hit
                result.retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
                result.initial_count = len(results_with_scores)

                # Set retrieval span attributes
                retrieval_span.set_attribute(SpanAttributes.RETRIEVAL_RESULT_COUNT, result.initial_count)
                retrieval_span.set_attribute(SpanAttributes.RETRIEVAL_TIME_MS, result.retrieval_time_ms)
                retrieval_span.set_attribute(SpanAttributes.RETRIEVAL_CACHE_HIT, cache_hit if cache_hit is not None else False)

            except Exception as e:
                result.retrieval_error = str(e)
                logger.error(f"Vector search failed: {e}")
                retrieval_span.record_exception(e)
                result.total_time_ms = (time.perf_counter() - total_start) * 1000
                return result

        if not results_with_scores:
            result.total_time_ms = (time.perf_counter() - total_start) * 1000
            return result

        # Extract documents and similarity scores
        documents = [doc for doc, _ in results_with_scores]
        similarity_scores = [score for _, score in results_with_scores]

        # Add score metrics to span after extraction
        if similarity_scores:
            avg_score = sum(similarity_scores) / len(similarity_scores)
            with tracer.start_as_current_span("rag.retrieval.scores") as scores_span:
                scores_span.set_attribute(SpanAttributes.RETRIEVAL_AVG_SCORE, avg_score)
                scores_span.set_attribute(SpanAttributes.RETRIEVAL_MAX_SCORE, max(similarity_scores))
                scores_span.set_attribute(SpanAttributes.RETRIEVAL_MIN_SCORE, min(similarity_scores))

        if settings.log_retrieval_details:
            score_preview = ", ".join(f"{s:.3f}" for s in similarity_scores[:5])
            logger.info(
                f"Vector search: {len(documents)} results in {result.retrieval_time_ms:.1f}ms, "
                f"scores=[{score_preview}]"
            )

        # Phase 2: Reranking (if enabled)
        if settings.reranker_enabled:
            with tracer.start_as_current_span("rag.rerank") as rerank_span:
                rerank_span.set_attribute(SpanAttributes.RERANK_ENABLED, True)
                rerank_span.set_attribute(SpanAttributes.RERANK_MODEL, settings.reranker_model)
                rerank_span.set_attribute(SpanAttributes.RERANK_INPUT_COUNT, len(documents))

                rerank_start = time.perf_counter()
                try:
                    reranked_docs = rerank_documents(
                        query=query,
                        documents=documents,
                        top_k=settings.reranker_top_k,
                    )
                    result.rerank_time_ms = (time.perf_counter() - rerank_start) * 1000
                    result.reranker_used = True
                    result.reranker_model = settings.reranker_model

                    # Set rerank span attributes
                    rerank_span.set_attribute(SpanAttributes.RERANK_OUTPUT_COUNT, len(reranked_docs))
                    rerank_span.set_attribute(SpanAttributes.RERANK_TIME_MS, result.rerank_time_ms)

                    # Extract rerank scores from document metadata
                    rerank_scores = []
                    for doc in reranked_docs:
                        score = doc.metadata.get('rerank_score', 0.0)
                        rerank_scores.append(score)

                    if rerank_scores:
                        rerank_span.set_attribute(SpanAttributes.RERANK_AVG_SCORE, sum(rerank_scores) / len(rerank_scores))

                    # Map similarity scores to reranked order using content hash
                    doc_to_sim_score = {}
                    for doc, score in results_with_scores:
                        key = hash(doc.page_content[:100])
                        doc_to_sim_score[key] = score

                    # Rebuild similarity scores in reranked order
                    new_similarity_scores = []
                    for doc in reranked_docs:
                        key = hash(doc.page_content[:100])
                        new_similarity_scores.append(doc_to_sim_score.get(key, 0.0))

                    documents = reranked_docs
                    similarity_scores = new_similarity_scores
                    result.rerank_scores = rerank_scores

                    if settings.log_retrieval_details:
                        rerank_preview = ", ".join(f"{s:.3f}" for s in rerank_scores[:5])
                        logger.info(
                            f"Reranking: {len(reranked_docs)} results in {result.rerank_time_ms:.1f}ms, "
                            f"scores=[{rerank_preview}]"
                        )

                    # Filter out documents with rerank scores below min_rerank_score threshold
                    filtered_docs = []
                    filtered_sim_scores = []
                    filtered_rerank_scores = []
                    for doc, sim_score, rerank_score in zip(documents, similarity_scores, rerank_scores):
                        if rerank_score >= settings.min_rerank_score:
                            filtered_docs.append(doc)
                            filtered_sim_scores.append(sim_score)
                            filtered_rerank_scores.append(rerank_score)

                    # Handle edge case: if all results filtered out, keep at least the top result
                    if not filtered_docs and documents:
                        filtered_docs = [documents[0]]
                        filtered_sim_scores = [similarity_scores[0]]
                        filtered_rerank_scores = [rerank_scores[0]]
                        logger.warning(
                            f"All {len(documents)} results filtered by min_rerank_score={settings.min_rerank_score}, "
                            f"keeping top result with score={rerank_scores[0]:.4f}"
                        )
                    elif len(filtered_docs) < len(documents):
                        filtered_count = len(documents) - len(filtered_docs)
                        if settings.log_retrieval_details:
                            logger.info(
                                f"Filtered {filtered_count} results below min_rerank_score={settings.min_rerank_score}"
                            )

                    documents = filtered_docs
                    similarity_scores = filtered_sim_scores
                    result.rerank_scores = filtered_rerank_scores

                except Exception as e:
                    result.rerank_error = str(e)
                    logger.error(f"Reranking failed: {e}, using original order")
                    rerank_span.record_exception(e)
                    # Fall back to original results without reranking
                    documents = documents[:settings.top_k_results]
                    similarity_scores = similarity_scores[:settings.top_k_results]
        else:
            # No reranking, just take top_k results
            documents = documents[:settings.top_k_results]
            similarity_scores = similarity_scores[:settings.top_k_results]

        result.documents = documents
        result.similarity_scores = similarity_scores
        result.final_count = len(documents)

        # Phase 3: Web search fallback (if enabled and local results are poor)
        if settings.web_search_enabled:
            # Use rerank scores if available (better quality signal than RRF similarity)
            if result.reranker_used and result.rerank_scores:
                avg_rerank = sum(result.rerank_scores) / len(result.rerank_scores)
                # Rerank scores: positive = relevant, negative = irrelevant
                # Trigger web search if avg rerank score is negative (poor relevance)
                if avg_rerank < 0:
                    should_search = True
                    reason = f"low_rerank_score_{avg_rerank:.2f}"
                else:
                    should_search = False
                    reason = None
            else:
                # Fallback to similarity scores if reranker not used
                avg_score = (
                    sum(result.similarity_scores) / len(result.similarity_scores)
                    if result.similarity_scores else 0.0
                )
                max_score = max(result.similarity_scores) if result.similarity_scores else 0.0

                should_search, reason = web_searcher.should_search(
                    avg_similarity_score=avg_score,
                    max_similarity_score=max_score,
                    result_count=result.final_count,
                )

            if should_search:
                result.web_search_trigger_reason = reason
                logger.info(f"Triggering web search fallback: {reason}")

                with tracer.start_as_current_span("rag.web_search") as web_span:
                    web_span.set_attribute(SpanAttributes.WEB_SEARCH_ENABLED, True)
                    web_span.set_attribute(SpanAttributes.WEB_SEARCH_TRIGGERED, True)
                    web_span.set_attribute(SpanAttributes.WEB_SEARCH_REASON, reason or "unknown")

                    try:
                        # Use synchronous search (httpx.Client, safe in any context)
                        web_response = web_searcher.search_sync(query)

                        result.web_search_time_ms = web_response.search_time_ms
                        result.web_search_used = web_response.triggered and not web_response.error

                        # Set span attributes
                        web_span.set_attribute(SpanAttributes.WEB_SEARCH_TIME_MS, result.web_search_time_ms)

                        if web_response.results:
                            result.web_search_results_count = len(web_response.results)
                            result.web_search_context = web_searcher.format_for_context(web_response.results)
                            web_span.set_attribute(SpanAttributes.WEB_SEARCH_RESULT_COUNT, result.web_search_results_count)
                            logger.info(
                                f"Web search returned {len(web_response.results)} results "
                                f"in {web_response.search_time_ms:.0f}ms"
                            )
                        elif web_response.error:
                            result.web_search_error = web_response.error
                            logger.warning(f"Web search error: {web_response.error}")

                    except Exception as e:
                        result.web_search_error = str(e)
                        logger.error(f"Web search fallback failed: {e}")
                        web_span.record_exception(e)

        result.total_time_ms = (time.perf_counter() - total_start) * 1000

        # Log metrics if enabled
        if settings.enable_retrieval_metrics:
            scores_to_log = result.rerank_scores if result.reranker_used else result.similarity_scores
            try:
                retrieval_metrics_logger.log_retrieval(
                    query=query,
                    model=model,
                    scores=scores_to_log,
                    top_k=settings.top_k_results,
                    latency_ms=result.total_time_ms,
                    score_threshold=settings.min_similarity_score,
                    filtered_count=0
                )
            except Exception as e:
                logger.warning(f"Failed to log retrieval metrics: {e}")

        # Record scores for drift detection (uses similarity scores for consistency)
        # This enables monitoring of embedding quality over time
        if result.similarity_scores:
            try:
                drift_detector.record_scores(result.similarity_scores)
            except Exception as e:
                logger.debug(f"Failed to record drift scores: {e}")

        return result

    async def _retrieve_with_scores_async(
        self,
        query: str,
        model: str = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> RetrievalResult:
        """Async wrapper for _retrieve_with_scores that runs in a thread pool executor.

        This prevents CPU-bound operations (vector search, reranking) from blocking
        the event loop in async contexts like FastAPI endpoints.

        Args:
            query: The search query string
            model: The LLM model being used (for metrics logging)
            conversation_history: Optional list of prior messages for context-aware retrieval

        Returns:
            RetrievalResult with documents, scores, timing, and metadata
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._retrieve_with_scores(query, model, conversation_history)
        )

    def _retrieve_with_metrics(
        self,
        query: str,
        model: str = None
    ) -> Tuple[List, Optional[RetrievalMetrics]]:
        """Backwards-compatible wrapper around _retrieve_with_scores.

        Returns documents and a simplified metrics object for existing code.
        """
        result = self._retrieve_with_scores(query, model)

        # Build simplified metrics for backwards compatibility
        metrics = None
        if settings.enable_retrieval_metrics and result.documents:
            scores = result.rerank_scores if result.reranker_used else result.similarity_scores
            if scores:
                import statistics
                metrics = RetrievalMetrics(
                    timestamp="",
                    query_hash="",
                    query_preview=query[:100],
                    model=model or self.default_model,
                    top_k=settings.top_k_results,
                    num_results=len(result.documents),
                    scores=scores,
                    latency_ms=result.total_time_ms,
                    score_threshold=settings.min_similarity_score,
                    filtered_count=0,
                    score_min=min(scores) if scores else None,
                    score_max=max(scores) if scores else None,
                    score_mean=statistics.mean(scores) if scores else None,
                )

        return result.documents, metrics

    def _retrieve_and_rerank(self, query: str) -> List:
        """Retrieve documents and optionally rerank them (backwards compatible).

        This is a convenience wrapper around _retrieve_with_scores that
        discards the detailed result for backwards compatibility.

        Args:
            query: The search query string

        Returns:
            List of Document objects, reranked if enabled
        """
        result = self._retrieve_with_scores(query)
        return result.documents

    def _format_sources_with_scores(self, result: RetrievalResult) -> List[Dict[str, Any]]:
        """Format retrieval results into source metadata with scores for API response.

        Each source includes:
        - source: Document source path/name
        - source_type: Type of documentation (kubernetes, terraform, etc.)
        - content_preview: First 200 chars of content
        - rank: Position in final results (1-indexed)
        - similarity_score: Vector similarity score (0-1)
        - rerank_score: Cross-encoder relevance score (if reranking used)

        Args:
            result: RetrievalResult from _retrieve_with_scores

        Returns:
            List of source dictionaries with scores and metadata
        """
        sources = []

        for i, doc in enumerate(result.documents):
            source_info = {
                'source': doc.metadata.get('source', 'Unknown'),
                'source_type': doc.metadata.get('source_type', 'Unknown'),
                'content_preview': (
                    doc.page_content[:200] + '...'
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
                'rank': i + 1,
            }

            # Add similarity score
            if i < len(result.similarity_scores):
                source_info['similarity_score'] = round(result.similarity_scores[i], 4)

            # Add rerank score if available
            if result.reranker_used and i < len(result.rerank_scores):
                source_info['rerank_score'] = round(result.rerank_scores[i], 4)

            sources.append(source_info)

        return sources

    def _build_retrieval_metrics_dict(self, result: RetrievalResult) -> Dict[str, Any]:
        """Build retrieval metrics dictionary for API response.

        Args:
            result: RetrievalResult from _retrieve_with_scores

        Returns:
            Dictionary with retrieval performance metrics
        """
        metrics = {
            'initial_candidates': result.initial_count,
            'final_results': result.final_count,
            'reranker_used': result.reranker_used,
            'reranker_model': result.reranker_model,
            'retrieval_time_ms': round(result.retrieval_time_ms, 2),
            'rerank_time_ms': round(result.rerank_time_ms, 2) if result.reranker_used else None,
            'total_time_ms': round(result.total_time_ms, 2),
            'hybrid_search_used': result.hybrid_search_used,
            'hyde_used': result.hyde_used,
            'hyde_time_ms': round(result.hyde_time_ms, 2) if result.hyde_used else None,
            'web_search_used': result.web_search_used,
            'web_search_reason': result.web_search_trigger_reason,
            'web_search_results': result.web_search_results_count if result.web_search_used else None,
            'web_search_time_ms': round(result.web_search_time_ms, 2) if result.web_search_used else None,
            'embedding_cache_hit': result.embedding_cache_hit,
            # Conversation context metrics
            'conversation_context_used': result.conversation_context_used,
            'conversation_context_terms': result.conversation_context_terms if result.conversation_context_used else None,
            'original_query': result.original_query if result.conversation_context_used else None,
        }

        # Calculate average scores
        if result.similarity_scores:
            metrics['avg_similarity_score'] = round(
                sum(result.similarity_scores) / len(result.similarity_scores), 4
            )

        if result.rerank_scores:
            metrics['avg_rerank_score'] = round(
                sum(result.rerank_scores) / len(result.rerank_scores), 4
            )

        # Include any errors
        if result.retrieval_error:
            metrics['retrieval_error'] = result.retrieval_error
        if result.rerank_error:
            metrics['rerank_error'] = result.rerank_error
        if result.web_search_error:
            metrics['web_search_error'] = result.web_search_error

        return metrics

    def get_reranker_status(self) -> Dict[str, Any]:
        """Get reranker component status for health checks.

        Returns:
            Dictionary with:
            - enabled: Whether reranker is enabled in config
            - loaded: Whether model is loaded in memory
            - model: Model name if loaded
            - device: Device (cpu/cuda) if loaded
            - error: Any error message
        """
        reranker = get_reranker()

        if reranker is None:
            return {
                'enabled': settings.reranker_enabled,
                'loaded': False,
                'model': None,
                'device': None,
                'error': None if not settings.reranker_enabled else 'Reranker not initialized'
            }

        return {
            'enabled': True,
            'loaded': True,
            **reranker.get_model_info()
        }

    def _validate_response(
        self,
        response: str,
        context: str = "",
        sources: Optional[List[Dict[str, Any]]] = None,
        query: str = ""
    ) -> ValidationResult:
        """Validate LLM response for hallucinations and quality issues.

        This method wraps the output_validation module to detect:
        - Hallucination markers (phrases indicating uncertainty/fabrication)
        - Unsupported claims (assertions not grounded in context)
        - Missing source citations
        - Fabrication patterns (overly specific details not from context)
        - Response length issues

        Args:
            response: The LLM-generated response text
            context: The context string provided to the LLM
            sources: List of source documents used for retrieval
            query: The original user query

        Returns:
            ValidationResult with issues, confidence score, and metadata
        """
        try:
            validation_result = validate_response(
                response=response,
                context=context,
                sources=sources,
                query=query
            )

            # Log validation summary
            if validation_result.issues:
                issue_summary = ", ".join(
                    f"{i.code}({i.severity.value})" for i in validation_result.issues[:5]
                )
                if len(validation_result.issues) > 5:
                    issue_summary += f"... (+{len(validation_result.issues) - 5} more)"

                logger.info(
                    f"Output validation: confidence={validation_result.confidence_score:.2f}, "
                    f"issues=[{issue_summary}], "
                    f"citations={validation_result.source_citation_count}, "
                    f"time={validation_result.validation_time_ms:.1f}ms"
                )

            return validation_result

        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            # Return a default result on error to not block the response
            return ValidationResult(
                is_valid=True,
                confidence_score=0.5,
                issues=[]
            )

    async def generate_response(
        self,
        query: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_rag: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Generate response using RAG pipeline with reranking and metrics.

        Flow:
            Query -> (Conversation Context Expansion) -> Vector Search (top 20) -> Rerank (top 5) -> Build Context -> LLM

        Args:
            query: User question/message
            model: Ollama model to use (default from settings)
            temperature: Generation temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            use_rag: Whether to use RAG context
            conversation_history: Optional list of prior messages for context-aware retrieval.
                                  Used to resolve pronouns and references in follow-up questions.

        Returns:
            Dictionary with:
            - response: LLM generated answer
            - model: Model used
            - context_used: Whether RAG context was used
            - sources: List of source documents with scores
            - retrieval_metrics: Performance and quality metrics (if enabled)
        """
        import time as time_module
        pipeline_start = time_module.perf_counter()
        tracer = get_tracer()

        # Compute query hash for tracing correlation
        query_hash = hashlib.md5(query.encode()).hexdigest()[:12]

        # Parent span for the entire RAG query
        with tracer.start_as_current_span("rag.query") as query_span:
            query_span.set_attribute(SpanAttributes.QUERY_HASH, query_hash)
            query_span.set_attribute(SpanAttributes.QUERY_LENGTH, len(query))
            query_span.set_attribute(SpanAttributes.LLM_MODEL, model or self.default_model)
            query_span.set_attribute(SpanAttributes.LLM_TEMPERATURE, temperature)
            query_span.set_attribute(SpanAttributes.LLM_MAX_TOKENS, max_tokens)

            model = model or self.default_model
            retrieval_result = RetrievalResult()
            context_str = ""

            # Retrieve relevant context if RAG is enabled (async to avoid blocking event loop)
            if use_rag:
                try:
                    retrieval_result = await self._retrieve_with_scores_async(
                        query, model, conversation_history
                    )
                    context_str = self._format_context(
                        retrieval_result.documents,
                        web_context=retrieval_result.web_search_context,
                        max_context_tokens=get_model_context_limit(model)
                    )
                except Exception as e:
                    logger.error("Error retrieving context: %s", e)
                    query_span.record_exception(e)

            # Build messages with proper system/user separation (model-specific prompt)
            messages = self._build_messages(query, context_str, model)

            # Set context attributes
            query_span.set_attribute(SpanAttributes.LLM_CONTEXT_LENGTH, len(context_str))
            query_span.set_attribute(SpanAttributes.PIPELINE_SOURCE_COUNT, len(retrieval_result.documents))

            # Generate response using Ollama with circuit breaker protection
            try:
                # LLM generation span
                with tracer.start_as_current_span("rag.llm_generate") as llm_span:
                    llm_span.set_attribute(SpanAttributes.LLM_MODEL, model)
                    llm_span.set_attribute(SpanAttributes.LLM_TEMPERATURE, temperature)
                    llm_span.set_attribute(SpanAttributes.LLM_MAX_TOKENS, max_tokens)
                    llm_span.set_attribute(SpanAttributes.LLM_CONTEXT_LENGTH, len(context_str))

                    llm_start = time_module.perf_counter()

                    # Wrap the Ollama call with circuit breaker
                    def _call_ollama():
                        return ollama.chat(
                            model=model,
                            messages=messages,
                            options={
                                'temperature': temperature,
                                'num_predict': max_tokens,
                            }
                        )

                    response = ollama_circuit_breaker.call(_call_ollama)

                    llm_time_ms = (time_module.perf_counter() - llm_start) * 1000
                    llm_span.set_attribute(SpanAttributes.LLM_TIME_MS, llm_time_ms)
                    llm_span.set_attribute(SpanAttributes.LLM_RESPONSE_LENGTH, len(response['message']['content']))

                answer = response['message']['content']

                # Build sources list for validation
                sources_list = (
                    self._format_sources_with_scores(retrieval_result)
                    if retrieval_result.documents else None
                )

                # Build response with sources including scores
                result = {
                    'response': answer,
                    'model': model,
                    'context_used': bool(retrieval_result.documents) or bool(retrieval_result.web_search_context),
                    'sources': sources_list,
                    'reranker_enabled': settings.reranker_enabled,
                }

                # Validate the response for hallucinations and quality issues (if enabled)
                if settings.output_validation_enabled:
                    validation_result = self._validate_response(
                        response=answer,
                        context=context_str,
                        sources=sources_list,
                        query=query
                    )
                    result['output_validation'] = validation_result.to_dict()

                # Include retrieval metrics if enabled
                if settings.enable_retrieval_metrics and (retrieval_result.documents or retrieval_result.web_search_used):
                    result['retrieval_metrics'] = self._build_retrieval_metrics_dict(retrieval_result)

                # Set final pipeline attributes
                pipeline_total_ms = (time_module.perf_counter() - pipeline_start) * 1000
                query_span.set_attribute(SpanAttributes.PIPELINE_TOTAL_TIME_MS, pipeline_total_ms)
                query_span.set_attribute(SpanAttributes.PIPELINE_CONTEXT_USED, result['context_used'])

                return result

            except CircuitBreakerOpen as e:
                # Provide user-friendly error for circuit breaker open
                query_span.record_exception(e)
                raise Exception(get_service_unavailable_message(e))
            except Exception as e:
                query_span.record_exception(e)
                raise Exception("Error generating response: {}".format(str(e)))
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models with circuit breaker protection."""
        try:
            # Use circuit breaker for Ollama list call
            def _list_models():
                return ollama.list()

            models = ollama_circuit_breaker.call(_list_models)
            return models.get('models', [])
        except CircuitBreakerOpen:
            logger.warning("Ollama circuit breaker is open, cannot list models")
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def is_ollama_connected(self) -> bool:
        """Check if Ollama is accessible (respects circuit breaker state)."""
        # Check circuit breaker state first
        cb_state = ollama_circuit_breaker.state
        if cb_state.value == "open":
            return False

        try:
            ollama.list()
            return True
        except:
            return False

    def _run_ollama_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Run synchronous Ollama streaming in a thread with circuit breaker protection.

        This method runs in a separate thread to avoid blocking the event loop.
        Uses circuit breaker to fail fast if Ollama is known to be unavailable.
        """
        try:
            # Check circuit breaker state before attempting stream
            cb_state = ollama_circuit_breaker.state
            if cb_state.value == "open":
                raise CircuitBreakerOpen(
                    "ollama",
                    ollama_circuit_breaker.config.reset_timeout_seconds
                )

            # Wrap streaming in circuit breaker call for failure tracking
            def _create_stream():
                return ollama.chat(
                    model=model,
                    messages=messages,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens,
                    },
                    stream=True
                )

            stream = ollama_circuit_breaker.call(_create_stream)

            for chunk in stream:
                if chunk.get('message', {}).get('content'):
                    # Thread-safe way to put item in async queue
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {'type': 'content', 'content': chunk['message']['content']}
                    )

            # Signal completion
            loop.call_soon_threadsafe(queue.put_nowait, {'type': 'done'})

        except CircuitBreakerOpen as e:
            error_msg = get_service_unavailable_message(e)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {'type': 'error', 'error': error_msg}
            )
        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {'type': 'error', 'error': str(e)}
            )

    async def generate_response_stream(
        self,
        query: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_rag: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ):
        """Generate streaming response using RAG pipeline with reranking.

        Flow:
            Query -> (Conversation Context Expansion) -> Vector Search (top 20) -> Rerank (top 5) -> Build Context -> LLM (streaming)

        This async generator properly yields control back to the event loop
        by running the synchronous Ollama streaming in a thread pool.

        Args:
            query: User question/message
            model: Ollama model to use (default from settings)
            temperature: Generation temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            use_rag: Whether to use RAG context
            conversation_history: Optional list of prior messages for context-aware retrieval

        Yields:
            Chunks with type field:
            - metadata: Initial response with sources, model, metrics
            - content: Token chunks from LLM
            - done: Completion signal
            - error: Error information if something fails
        """
        import time as time_module
        tracer = get_tracer()

        # Compute query hash for tracing correlation
        query_hash = hashlib.md5(query.encode()).hexdigest()[:12]

        model = model or self.default_model
        retrieval_result = RetrievalResult()
        context_str = ""

        # Create parent span for streaming query (note: spans don't work well with async generators,
        # so we create events instead of nested spans for streaming)
        with tracer.start_as_current_span("rag.query.stream") as query_span:
            query_span.set_attribute(SpanAttributes.QUERY_HASH, query_hash)
            query_span.set_attribute(SpanAttributes.QUERY_LENGTH, len(query))
            query_span.set_attribute(SpanAttributes.LLM_MODEL, model)

            # Retrieve relevant context if RAG is enabled (async to avoid blocking event loop)
            if use_rag:
                try:
                    retrieval_result = await self._retrieve_with_scores_async(
                        query, model, conversation_history
                    )
                    context_str = self._format_context(
                        retrieval_result.documents,
                        web_context=retrieval_result.web_search_context,
                        max_context_tokens=get_model_context_limit(model)
                    )
                    query_span.set_attribute(SpanAttributes.PIPELINE_SOURCE_COUNT, len(retrieval_result.documents))
                except Exception as e:
                    logger.error("Error retrieving context: %s", e)
                    query_span.record_exception(e)

            # Build messages with proper system/user separation (model-specific prompt)
            messages = self._build_messages(query, context_str, model)

            query_span.set_attribute(SpanAttributes.LLM_CONTEXT_LENGTH, len(context_str))

        # Build metadata response with sources including scores
        metadata = {
            'type': 'metadata',
            'model': model,
            'context_used': bool(retrieval_result.documents) or bool(retrieval_result.web_search_context),
            'sources': (
                self._format_sources_with_scores(retrieval_result)
                if retrieval_result.documents else None
            ),
            'reranker_enabled': settings.reranker_enabled,
        }

        # Include retrieval metrics if enabled
        if settings.enable_retrieval_metrics and retrieval_result.documents:
            metadata['retrieval_metrics'] = self._build_retrieval_metrics_dict(retrieval_result)

        # Yield metadata first
        yield metadata

        # Generate streaming response using Ollama in a thread pool
        # This prevents blocking the event loop
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Run the synchronous Ollama streaming in a thread pool
        thread_task = loop.run_in_executor(
            None,  # Use default executor (ThreadPoolExecutor)
            self._run_ollama_stream,
            model,
            messages,
            temperature,
            max_tokens,
            queue,
            loop,
        )

        # Consume chunks from the queue as they arrive, accumulating response for validation
        accumulated_response = []
        try:
            while True:
                # Wait for next chunk from the thread, yielding control to event loop
                chunk = await queue.get()

                if chunk['type'] == 'done':
                    # Validate the complete response before signaling done (if enabled)
                    if settings.output_validation_enabled:
                        full_response = "".join(accumulated_response)
                        sources_list = metadata.get('sources')

                        validation_result = self._validate_response(
                            response=full_response,
                            context=context_str,
                            sources=sources_list,
                            query=query
                        )

                        # Yield validation results before done signal
                        yield {
                            'type': 'validation',
                            'output_validation': validation_result.to_dict()
                        }

                    yield chunk
                    break
                elif chunk['type'] == 'error':
                    yield chunk
                    break
                else:
                    # Accumulate content for validation
                    if chunk.get('content'):
                        accumulated_response.append(chunk['content'])
                    yield chunk
        finally:
            # Ensure the thread task completes
            await thread_task


# Singleton instance
rag_pipeline = RAGPipeline()
