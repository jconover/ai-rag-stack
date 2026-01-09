"""RAG Evaluation System for measuring retrieval quality.

This module provides a comprehensive evaluation framework for the RAG pipeline:
- Ground truth dataset structure with expected relevant documents and keywords
- Standard IR metrics: MRR@k, Recall@k, Precision@k
- Evaluation runner for systematic testing
- Support for multiple documentation categories

Metrics implemented:
- Mean Reciprocal Rank (MRR@k): Position of first relevant result
- Recall@k: Fraction of relevant docs retrieved
- Precision@k: Fraction of retrieved docs that are relevant
- Average Similarity Score: Mean vector similarity of retrieved docs

Usage:
    from app.evaluation import run_evaluation_suite, SEED_EVAL_DATASET

    # Run full evaluation
    results = await run_evaluation_suite()

    # Run evaluation on specific category
    results = await run_evaluation_suite(category="kubernetes")
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import json

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures for Evaluation
# =============================================================================

@dataclass
class EvalExample:
    """A single evaluation example with ground truth labels.

    Attributes:
        query: The search query to evaluate
        relevant_doc_ids: List of expected document source identifiers (substrings to match)
        expected_keywords: Keywords that should appear in relevant results
        category: Documentation category (kubernetes, terraform, docker, etc.)
        difficulty: Optional difficulty level (easy, medium, hard)
        notes: Optional notes about this example
    """
    query: str
    relevant_doc_ids: List[str]
    expected_keywords: List[str]
    category: str
    difficulty: str = "medium"
    notes: Optional[str] = None


@dataclass
class RAGEvalMetrics:
    """Aggregated evaluation metrics for a single query or batch.

    Attributes:
        mrr_at_k: Mean Reciprocal Rank - position of first relevant doc (1/rank)
        recall_at_k: Fraction of relevant docs that were retrieved
        precision_at_k: Fraction of retrieved docs that are relevant
        avg_similarity_score: Average vector similarity score of retrieved docs
        ndcg_at_k: Normalized Discounted Cumulative Gain (optional)
        keyword_hit_rate: Fraction of expected keywords found in results
    """
    mrr_at_k: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    avg_similarity_score: float = 0.0
    ndcg_at_k: Optional[float] = None
    keyword_hit_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return asdict(self)


@dataclass
class EvalResult:
    """Result of evaluating a single query.

    Attributes:
        example: The evaluation example that was tested
        metrics: Computed metrics for this query
        retrieved_sources: List of source identifiers from retrieved docs
        retrieved_scores: Similarity scores for retrieved docs
        relevant_found: List of relevant doc IDs that were found
        relevant_missed: List of relevant doc IDs that were not found
        keywords_found: Keywords from expected list that were found
        keywords_missed: Keywords from expected list that were not found
        retrieval_time_ms: Time taken for retrieval in milliseconds
        error: Error message if evaluation failed
    """
    example: EvalExample
    metrics: RAGEvalMetrics
    retrieved_sources: List[str] = field(default_factory=list)
    retrieved_scores: List[float] = field(default_factory=list)
    relevant_found: List[str] = field(default_factory=list)
    relevant_missed: List[str] = field(default_factory=list)
    keywords_found: List[str] = field(default_factory=list)
    keywords_missed: List[str] = field(default_factory=list)
    retrieval_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class EvalSuiteResult:
    """Aggregated results from running the full evaluation suite.

    Attributes:
        total_examples: Number of examples evaluated
        successful_evals: Number of successful evaluations
        failed_evals: Number of failed evaluations
        avg_mrr: Average MRR across all examples
        avg_recall: Average Recall@k across all examples
        avg_precision: Average Precision@k across all examples
        avg_similarity: Average similarity score across all examples
        avg_keyword_hit_rate: Average keyword hit rate across all examples
        avg_retrieval_time_ms: Average retrieval time in milliseconds
        by_category: Metrics broken down by category
        by_difficulty: Metrics broken down by difficulty
        individual_results: List of all individual evaluation results
        timestamp: When the evaluation was run
        config: Configuration used for evaluation
    """
    total_examples: int = 0
    successful_evals: int = 0
    failed_evals: int = 0
    avg_mrr: float = 0.0
    avg_recall: float = 0.0
    avg_precision: float = 0.0
    avg_similarity: float = 0.0
    avg_keyword_hit_rate: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    individual_results: List[EvalResult] = field(default_factory=list)
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding individual results for summary."""
        result = {
            "total_examples": self.total_examples,
            "successful_evals": self.successful_evals,
            "failed_evals": self.failed_evals,
            "avg_mrr": round(self.avg_mrr, 4),
            "avg_recall": round(self.avg_recall, 4),
            "avg_precision": round(self.avg_precision, 4),
            "avg_similarity": round(self.avg_similarity, 4),
            "avg_keyword_hit_rate": round(self.avg_keyword_hit_rate, 4),
            "avg_retrieval_time_ms": round(self.avg_retrieval_time_ms, 2),
            "by_category": self.by_category,
            "by_difficulty": self.by_difficulty,
            "timestamp": self.timestamp,
            "config": self.config,
        }
        return result

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Seed Evaluation Dataset
# =============================================================================

SEED_EVAL_DATASET: List[EvalExample] = [
    # --- Kubernetes Queries ---
    EvalExample(
        query="How do I create a Kubernetes deployment?",
        relevant_doc_ids=["kubernetes", "deployment", "kubectl"],
        expected_keywords=["deployment", "replicas", "spec", "template", "kubectl apply"],
        category="kubernetes",
        difficulty="easy",
        notes="Basic deployment creation question"
    ),
    EvalExample(
        query="What is a Kubernetes pod and how does it differ from a container?",
        relevant_doc_ids=["kubernetes", "pod", "container"],
        expected_keywords=["pod", "container", "shared", "network", "volume"],
        category="kubernetes",
        difficulty="easy",
        notes="Fundamental concept question"
    ),
    EvalExample(
        query="How do I expose a Kubernetes service externally using LoadBalancer?",
        relevant_doc_ids=["kubernetes", "service", "loadbalancer"],
        expected_keywords=["service", "LoadBalancer", "type", "port", "external"],
        category="kubernetes",
        difficulty="medium",
        notes="Service exposure question"
    ),
    EvalExample(
        query="How do I configure horizontal pod autoscaling based on CPU usage?",
        relevant_doc_ids=["kubernetes", "autoscal", "hpa"],
        expected_keywords=["HorizontalPodAutoscaler", "CPU", "minReplicas", "maxReplicas", "metrics"],
        category="kubernetes",
        difficulty="medium",
        notes="Autoscaling configuration"
    ),

    # --- Docker Queries ---
    EvalExample(
        query="How do I build a Docker image from a Dockerfile?",
        relevant_doc_ids=["docker", "dockerfile", "build"],
        expected_keywords=["docker build", "Dockerfile", "-t", "tag", "image"],
        category="docker",
        difficulty="easy",
        notes="Basic Docker build question"
    ),
    EvalExample(
        query="How do I set up Docker Compose for multi-container applications?",
        relevant_doc_ids=["docker", "compose", "multi-container"],
        expected_keywords=["docker-compose", "services", "volumes", "networks", "yaml"],
        category="docker",
        difficulty="medium",
        notes="Docker Compose setup"
    ),
    EvalExample(
        query="What are Docker networking modes and when to use each?",
        relevant_doc_ids=["docker", "network", "bridge", "host"],
        expected_keywords=["bridge", "host", "none", "overlay", "network"],
        category="docker",
        difficulty="medium",
        notes="Docker networking concepts"
    ),

    # --- Terraform Queries ---
    EvalExample(
        query="How do I manage Terraform state in a team environment?",
        relevant_doc_ids=["terraform", "state", "backend", "remote"],
        expected_keywords=["state", "backend", "remote", "s3", "locking"],
        category="terraform",
        difficulty="medium",
        notes="Terraform state management"
    ),
    EvalExample(
        query="What are Terraform modules and how do I create one?",
        relevant_doc_ids=["terraform", "module", "reusable"],
        expected_keywords=["module", "source", "variable", "output", "reusable"],
        category="terraform",
        difficulty="medium",
        notes="Terraform modules question"
    ),
    EvalExample(
        query="How do I use Terraform providers for AWS resources?",
        relevant_doc_ids=["terraform", "provider", "aws"],
        expected_keywords=["provider", "aws", "region", "resource", "terraform init"],
        category="terraform",
        difficulty="easy",
        notes="Provider configuration"
    ),

    # --- CI/CD Queries ---
    EvalExample(
        query="How do I set up a GitHub Actions workflow for CI/CD?",
        relevant_doc_ids=["github", "actions", "workflow", "ci"],
        expected_keywords=["workflow", "jobs", "steps", "on", "runs-on", "yaml"],
        category="cicd",
        difficulty="medium",
        notes="GitHub Actions setup"
    ),
    EvalExample(
        query="What are GitHub Actions secrets and how do I use them?",
        relevant_doc_ids=["github", "actions", "secrets", "environment"],
        expected_keywords=["secrets", "env", "GITHUB_TOKEN", "encrypted"],
        category="cicd",
        difficulty="easy",
        notes="Secrets management in CI"
    ),

    # --- Ansible Queries ---
    EvalExample(
        query="How do I write an Ansible playbook for server configuration?",
        relevant_doc_ids=["ansible", "playbook", "task", "yaml"],
        expected_keywords=["playbook", "hosts", "tasks", "module", "yaml"],
        category="ansible",
        difficulty="medium",
        notes="Basic Ansible playbook"
    ),

    # --- Prometheus/Monitoring Queries ---
    EvalExample(
        query="How do I configure Prometheus alerting rules?",
        relevant_doc_ids=["prometheus", "alert", "rule"],
        expected_keywords=["alert", "rules", "expr", "for", "labels", "annotations"],
        category="monitoring",
        difficulty="medium",
        notes="Prometheus alerting"
    ),

    # --- Helm Queries ---
    EvalExample(
        query="How do I create a Helm chart for my application?",
        relevant_doc_ids=["helm", "chart", "template"],
        expected_keywords=["chart", "values", "templates", "Chart.yaml", "helm create"],
        category="helm",
        difficulty="medium",
        notes="Helm chart creation"
    ),
]


# =============================================================================
# Core Evaluation Functions
# =============================================================================

def calculate_mrr(
    retrieved_docs: List[Tuple[Any, float]],
    relevant_doc_ids: List[str],
    k: int = 5
) -> float:
    """Calculate Mean Reciprocal Rank at k.

    MRR measures the position of the first relevant document.
    MRR = 1/rank where rank is the position of the first relevant doc.

    Args:
        retrieved_docs: List of (Document, score) tuples from retrieval
        relevant_doc_ids: List of substrings that identify relevant documents
        k: Number of results to consider

    Returns:
        MRR score (0.0 to 1.0, higher is better)
    """
    for rank, (doc, _score) in enumerate(retrieved_docs[:k], start=1):
        source = doc.metadata.get('source', '').lower()
        source_type = doc.metadata.get('source_type', '').lower()
        content = doc.page_content.lower()

        # Check if any relevant doc ID matches this document
        for relevant_id in relevant_doc_ids:
            relevant_lower = relevant_id.lower()
            if (relevant_lower in source or
                relevant_lower in source_type or
                relevant_lower in content[:500]):  # Check beginning of content
                return 1.0 / rank

    return 0.0  # No relevant document found in top k


def calculate_recall(
    retrieved_docs: List[Tuple[Any, float]],
    relevant_doc_ids: List[str],
    k: int = 5
) -> float:
    """Calculate Recall at k.

    Recall measures the fraction of relevant documents that were retrieved.
    Recall@k = |relevant docs retrieved| / |total relevant docs|

    Args:
        retrieved_docs: List of (Document, score) tuples from retrieval
        relevant_doc_ids: List of substrings that identify relevant documents
        k: Number of results to consider

    Returns:
        Recall score (0.0 to 1.0, higher is better)
    """
    if not relevant_doc_ids:
        return 1.0  # No relevant docs expected, consider it full recall

    found_relevant = set()

    for doc, _score in retrieved_docs[:k]:
        source = doc.metadata.get('source', '').lower()
        source_type = doc.metadata.get('source_type', '').lower()
        content = doc.page_content.lower()

        for relevant_id in relevant_doc_ids:
            relevant_lower = relevant_id.lower()
            if (relevant_lower in source or
                relevant_lower in source_type or
                relevant_lower in content[:500]):
                found_relevant.add(relevant_id)

    return len(found_relevant) / len(relevant_doc_ids)


def calculate_precision(
    retrieved_docs: List[Tuple[Any, float]],
    relevant_doc_ids: List[str],
    k: int = 5
) -> float:
    """Calculate Precision at k.

    Precision measures the fraction of retrieved documents that are relevant.
    Precision@k = |relevant docs in top k| / k

    Args:
        retrieved_docs: List of (Document, score) tuples from retrieval
        relevant_doc_ids: List of substrings that identify relevant documents
        k: Number of results to consider

    Returns:
        Precision score (0.0 to 1.0, higher is better)
    """
    if not retrieved_docs:
        return 0.0

    relevant_count = 0
    actual_k = min(k, len(retrieved_docs))

    for doc, _score in retrieved_docs[:actual_k]:
        source = doc.metadata.get('source', '').lower()
        source_type = doc.metadata.get('source_type', '').lower()
        content = doc.page_content.lower()

        is_relevant = False
        for relevant_id in relevant_doc_ids:
            relevant_lower = relevant_id.lower()
            if (relevant_lower in source or
                relevant_lower in source_type or
                relevant_lower in content[:500]):
                is_relevant = True
                break

        if is_relevant:
            relevant_count += 1

    return relevant_count / actual_k


def calculate_keyword_hit_rate(
    retrieved_docs: List[Tuple[Any, float]],
    expected_keywords: List[str],
    k: int = 5
) -> Tuple[float, List[str], List[str]]:
    """Calculate keyword hit rate in retrieved documents.

    Args:
        retrieved_docs: List of (Document, score) tuples from retrieval
        expected_keywords: Keywords expected to appear in relevant results
        k: Number of results to consider

    Returns:
        Tuple of (hit_rate, keywords_found, keywords_missed)
    """
    if not expected_keywords:
        return 1.0, [], []

    # Combine all content from top k documents
    combined_content = ""
    for doc, _score in retrieved_docs[:k]:
        combined_content += " " + doc.page_content.lower()
        combined_content += " " + doc.metadata.get('source', '').lower()

    keywords_found = []
    keywords_missed = []

    for keyword in expected_keywords:
        if keyword.lower() in combined_content:
            keywords_found.append(keyword)
        else:
            keywords_missed.append(keyword)

    hit_rate = len(keywords_found) / len(expected_keywords)
    return hit_rate, keywords_found, keywords_missed


def calculate_avg_similarity(
    retrieved_docs: List[Tuple[Any, float]],
    k: int = 5
) -> float:
    """Calculate average similarity score of retrieved documents.

    Args:
        retrieved_docs: List of (Document, score) tuples from retrieval
        k: Number of results to consider

    Returns:
        Average similarity score (0.0 to 1.0)
    """
    if not retrieved_docs:
        return 0.0

    scores = [score for _, score in retrieved_docs[:k]]
    return sum(scores) / len(scores)


def evaluate_retrieval(
    query: str,
    retrieved_docs: List[Tuple[Any, float]],
    eval_example: EvalExample,
    k: int = 5
) -> EvalResult:
    """Run all evaluation metrics for a single query.

    Args:
        query: The query that was used for retrieval
        retrieved_docs: List of (Document, score) tuples from vector store
        eval_example: Ground truth evaluation example
        k: Number of results to consider

    Returns:
        EvalResult with all computed metrics
    """
    # Calculate core metrics
    mrr = calculate_mrr(retrieved_docs, eval_example.relevant_doc_ids, k)
    recall = calculate_recall(retrieved_docs, eval_example.relevant_doc_ids, k)
    precision = calculate_precision(retrieved_docs, eval_example.relevant_doc_ids, k)
    avg_sim = calculate_avg_similarity(retrieved_docs, k)
    keyword_rate, kw_found, kw_missed = calculate_keyword_hit_rate(
        retrieved_docs, eval_example.expected_keywords, k
    )

    metrics = RAGEvalMetrics(
        mrr_at_k=mrr,
        recall_at_k=recall,
        precision_at_k=precision,
        avg_similarity_score=avg_sim,
        keyword_hit_rate=keyword_rate,
    )

    # Determine which relevant docs were found/missed
    found_relevant = []
    missed_relevant = list(eval_example.relevant_doc_ids)

    for doc, _score in retrieved_docs[:k]:
        source = doc.metadata.get('source', '').lower()
        source_type = doc.metadata.get('source_type', '').lower()
        content = doc.page_content.lower()

        for relevant_id in eval_example.relevant_doc_ids:
            relevant_lower = relevant_id.lower()
            if (relevant_lower in source or
                relevant_lower in source_type or
                relevant_lower in content[:500]):
                if relevant_id not in found_relevant:
                    found_relevant.append(relevant_id)
                if relevant_id in missed_relevant:
                    missed_relevant.remove(relevant_id)

    # Extract retrieved sources and scores
    retrieved_sources = [doc.metadata.get('source', 'unknown') for doc, _ in retrieved_docs[:k]]
    retrieved_scores = [score for _, score in retrieved_docs[:k]]

    return EvalResult(
        example=eval_example,
        metrics=metrics,
        retrieved_sources=retrieved_sources,
        retrieved_scores=retrieved_scores,
        relevant_found=found_relevant,
        relevant_missed=missed_relevant,
        keywords_found=kw_found,
        keywords_missed=kw_missed,
    )


# =============================================================================
# Evaluation Runner
# =============================================================================

async def run_evaluation_suite(
    dataset: Optional[List[EvalExample]] = None,
    category: Optional[str] = None,
    k: int = None,
    min_score: float = None,
    use_hybrid: Optional[bool] = None,
    use_reranker: Optional[bool] = None,
) -> EvalSuiteResult:
    """Run the full evaluation suite on the dataset.

    This function evaluates retrieval quality by running each example
    through the vector store and computing standard IR metrics.

    Args:
        dataset: Evaluation dataset to use (defaults to SEED_EVAL_DATASET)
        category: Filter to only evaluate specific category (e.g., "kubernetes")
        k: Number of results to retrieve (defaults to settings.top_k_results)
        min_score: Minimum similarity score threshold
        use_hybrid: Whether to use hybrid search (defaults to settings)
        use_reranker: Whether to use reranker (defaults to settings)

    Returns:
        EvalSuiteResult with aggregated metrics and individual results
    """
    # Import here to avoid circular imports
    from app.vectorstore import vector_store

    # Use defaults from settings if not specified
    if k is None:
        k = settings.top_k_results
    if min_score is None:
        min_score = settings.min_similarity_score
    if use_hybrid is None:
        use_hybrid = settings.hybrid_search_enabled
    if use_reranker is None:
        use_reranker = settings.reranker_enabled

    # Select dataset
    if dataset is None:
        dataset = SEED_EVAL_DATASET

    # Filter by category if specified
    if category:
        dataset = [ex for ex in dataset if ex.category.lower() == category.lower()]

    if not dataset:
        logger.warning("No evaluation examples found for specified criteria")
        return EvalSuiteResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            config={"error": "No examples found"}
        )

    # Run evaluations
    results: List[EvalResult] = []
    total_time_ms = 0.0

    for example in dataset:
        start_time = time.perf_counter()
        try:
            # Perform retrieval
            if use_hybrid:
                retrieved_docs = vector_store.hybrid_search_with_scores(
                    query=example.query,
                    top_k=k if not use_reranker else settings.retrieval_top_k,
                    min_score=min_score,
                )
            else:
                retrieved_docs = vector_store.search_with_scores(
                    query=example.query,
                    top_k=k if not use_reranker else settings.retrieval_top_k,
                    min_score=min_score,
                )

            # Apply reranking if enabled
            if use_reranker and retrieved_docs:
                from app.reranker import rerank_documents
                docs_only = [doc for doc, _ in retrieved_docs]
                reranked = rerank_documents(
                    query=example.query,
                    documents=docs_only,
                    top_k=k,
                )
                # Rebuild docs with scores (use rerank scores from metadata)
                retrieved_docs = [
                    (doc, doc.metadata.get('rerank_score', 0.0))
                    for doc in reranked
                ]

            retrieval_time = (time.perf_counter() - start_time) * 1000
            total_time_ms += retrieval_time

            # Evaluate
            eval_result = evaluate_retrieval(
                query=example.query,
                retrieved_docs=retrieved_docs,
                eval_example=example,
                k=k,
            )
            eval_result.retrieval_time_ms = retrieval_time
            results.append(eval_result)

        except Exception as e:
            logger.error(f"Evaluation failed for query '{example.query[:50]}...': {e}")
            results.append(EvalResult(
                example=example,
                metrics=RAGEvalMetrics(),
                error=str(e),
                retrieval_time_ms=(time.perf_counter() - start_time) * 1000,
            ))

    # Aggregate results
    successful_results = [r for r in results if r.error is None]

    suite_result = EvalSuiteResult(
        total_examples=len(results),
        successful_evals=len(successful_results),
        failed_evals=len(results) - len(successful_results),
        individual_results=results,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config={
            "k": k,
            "min_score": min_score,
            "use_hybrid": use_hybrid,
            "use_reranker": use_reranker,
            "embedding_model": settings.embedding_model,
            "category_filter": category,
        }
    )

    if successful_results:
        suite_result.avg_mrr = sum(r.metrics.mrr_at_k for r in successful_results) / len(successful_results)
        suite_result.avg_recall = sum(r.metrics.recall_at_k for r in successful_results) / len(successful_results)
        suite_result.avg_precision = sum(r.metrics.precision_at_k for r in successful_results) / len(successful_results)
        suite_result.avg_similarity = sum(r.metrics.avg_similarity_score for r in successful_results) / len(successful_results)
        suite_result.avg_keyword_hit_rate = sum(r.metrics.keyword_hit_rate for r in successful_results) / len(successful_results)
        suite_result.avg_retrieval_time_ms = total_time_ms / len(results)

        # Compute by-category metrics
        categories = set(r.example.category for r in successful_results)
        for cat in categories:
            cat_results = [r for r in successful_results if r.example.category == cat]
            suite_result.by_category[cat] = {
                "count": len(cat_results),
                "avg_mrr": round(sum(r.metrics.mrr_at_k for r in cat_results) / len(cat_results), 4),
                "avg_recall": round(sum(r.metrics.recall_at_k for r in cat_results) / len(cat_results), 4),
                "avg_precision": round(sum(r.metrics.precision_at_k for r in cat_results) / len(cat_results), 4),
            }

        # Compute by-difficulty metrics
        difficulties = set(r.example.difficulty for r in successful_results)
        for diff in difficulties:
            diff_results = [r for r in successful_results if r.example.difficulty == diff]
            suite_result.by_difficulty[diff] = {
                "count": len(diff_results),
                "avg_mrr": round(sum(r.metrics.mrr_at_k for r in diff_results) / len(diff_results), 4),
                "avg_recall": round(sum(r.metrics.recall_at_k for r in diff_results) / len(diff_results), 4),
                "avg_precision": round(sum(r.metrics.precision_at_k for r in diff_results) / len(diff_results), 4),
            }

    return suite_result


def run_evaluation_suite_sync(
    dataset: Optional[List[EvalExample]] = None,
    category: Optional[str] = None,
    k: int = None,
    min_score: float = None,
    use_hybrid: Optional[bool] = None,
    use_reranker: Optional[bool] = None,
) -> EvalSuiteResult:
    """Synchronous wrapper for run_evaluation_suite.

    Useful for running from non-async contexts like scripts or tests.
    """
    return asyncio.get_event_loop().run_until_complete(
        run_evaluation_suite(
            dataset=dataset,
            category=category,
            k=k,
            min_score=min_score,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )
    )


# =============================================================================
# Utility Functions
# =============================================================================

def get_evaluation_categories() -> List[str]:
    """Get list of unique categories in the seed dataset."""
    return sorted(set(ex.category for ex in SEED_EVAL_DATASET))


def get_evaluation_stats() -> Dict[str, Any]:
    """Get statistics about the evaluation dataset."""
    categories = {}
    difficulties = {}

    for ex in SEED_EVAL_DATASET:
        categories[ex.category] = categories.get(ex.category, 0) + 1
        difficulties[ex.difficulty] = difficulties.get(ex.difficulty, 0) + 1

    return {
        "total_examples": len(SEED_EVAL_DATASET),
        "categories": categories,
        "difficulties": difficulties,
    }


def add_eval_example(example: EvalExample) -> None:
    """Add a new example to the seed dataset (in-memory only).

    For persistent storage, the example should be added to SEED_EVAL_DATASET
    in the source code or stored in a database.
    """
    SEED_EVAL_DATASET.append(example)
    logger.info(f"Added evaluation example: {example.query[:50]}...")


def format_eval_report(result: EvalSuiteResult) -> str:
    """Format evaluation results as a human-readable report.

    Args:
        result: EvalSuiteResult from run_evaluation_suite

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "RAG EVALUATION REPORT",
        "=" * 60,
        f"Timestamp: {result.timestamp}",
        f"Total Examples: {result.total_examples}",
        f"Successful: {result.successful_evals}",
        f"Failed: {result.failed_evals}",
        "",
        "AGGREGATE METRICS",
        "-" * 40,
        f"Mean Reciprocal Rank (MRR@k): {result.avg_mrr:.4f}",
        f"Recall@k:                     {result.avg_recall:.4f}",
        f"Precision@k:                  {result.avg_precision:.4f}",
        f"Average Similarity Score:     {result.avg_similarity:.4f}",
        f"Keyword Hit Rate:             {result.avg_keyword_hit_rate:.4f}",
        f"Average Retrieval Time:       {result.avg_retrieval_time_ms:.2f} ms",
        "",
        "CONFIGURATION",
        "-" * 40,
    ]

    for key, value in result.config.items():
        lines.append(f"  {key}: {value}")

    if result.by_category:
        lines.extend([
            "",
            "METRICS BY CATEGORY",
            "-" * 40,
        ])
        for cat, metrics in sorted(result.by_category.items()):
            lines.append(f"  {cat} (n={metrics['count']}): "
                        f"MRR={metrics['avg_mrr']:.3f}, "
                        f"Recall={metrics['avg_recall']:.3f}, "
                        f"Precision={metrics['avg_precision']:.3f}")

    if result.by_difficulty:
        lines.extend([
            "",
            "METRICS BY DIFFICULTY",
            "-" * 40,
        ])
        for diff, metrics in sorted(result.by_difficulty.items()):
            lines.append(f"  {diff} (n={metrics['count']}): "
                        f"MRR={metrics['avg_mrr']:.3f}, "
                        f"Recall={metrics['avg_recall']:.3f}, "
                        f"Precision={metrics['avg_precision']:.3f}")

    # Add failed examples if any
    failed = [r for r in result.individual_results if r.error]
    if failed:
        lines.extend([
            "",
            "FAILED EVALUATIONS",
            "-" * 40,
        ])
        for r in failed:
            lines.append(f"  - {r.example.query[:50]}...: {r.error}")

    lines.append("=" * 60)

    return "\n".join(lines)
