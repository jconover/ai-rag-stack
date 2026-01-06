#!/usr/bin/env python3
"""
RAG System Benchmarking Framework

This script benchmarks the RAG system's retrieval and response performance
across various configurations. It measures:
- Retrieval latency (vector search time)
- Reranking latency (if enabled)
- Total response time
- Retrieval quality scores

Usage:
    python scripts/benchmark_rag.py [--output results.json] [--skip-llm]

Requirements:
    pip install numpy scipy tabulate tqdm
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import statistics

import numpy as np
from scipy import stats as scipy_stats
from tabulate import tabulate
from tqdm import tqdm

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "devops_docs")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# =============================================================================
# Test Query Definitions
# =============================================================================

@dataclass
class TestQuery:
    """Represents a test query with metadata for evaluation"""
    query: str
    category: str
    complexity: str  # "simple", "medium", "complex"
    expected_sources: List[str]  # Expected source_type values
    description: str = ""


# 20 DevOps test queries across categories
TEST_QUERIES: List[TestQuery] = [
    # Kubernetes - Basic (4 queries)
    TestQuery(
        query="How do I create a Kubernetes deployment?",
        category="kubernetes",
        complexity="simple",
        expected_sources=["kubernetes"],
        description="Basic K8s deployment creation"
    ),
    TestQuery(
        query="What is the difference between a Deployment and a StatefulSet in Kubernetes?",
        category="kubernetes",
        complexity="medium",
        expected_sources=["kubernetes"],
        description="K8s workload comparison"
    ),
    TestQuery(
        query="How do I configure horizontal pod autoscaling with custom metrics?",
        category="kubernetes",
        complexity="complex",
        expected_sources=["kubernetes"],
        description="Advanced HPA configuration"
    ),
    TestQuery(
        query="Explain Kubernetes service types and when to use each",
        category="kubernetes",
        complexity="medium",
        expected_sources=["kubernetes"],
        description="K8s networking concepts"
    ),

    # Kubernetes AI/ML (3 queries)
    TestQuery(
        query="How do I deploy a GPU workload on Kubernetes?",
        category="kubernetes-ai",
        complexity="medium",
        expected_sources=["kubernetes-ai", "kubernetes"],
        description="GPU scheduling on K8s"
    ),
    TestQuery(
        query="What is the Kubernetes scheduler and how does it work for ML workloads?",
        category="kubernetes-ai",
        complexity="complex",
        expected_sources=["kubernetes-ai", "kubernetes"],
        description="K8s scheduling for AI/ML"
    ),
    TestQuery(
        query="How do I set up distributed training on Kubernetes?",
        category="kubernetes-ai",
        complexity="complex",
        expected_sources=["kubernetes-ai", "kubernetes"],
        description="Distributed ML training"
    ),

    # Terraform (4 queries)
    TestQuery(
        query="How do I create an AWS EC2 instance with Terraform?",
        category="terraform",
        complexity="simple",
        expected_sources=["terraform"],
        description="Basic Terraform resource"
    ),
    TestQuery(
        query="What are Terraform modules and how do I use them?",
        category="terraform",
        complexity="medium",
        expected_sources=["terraform"],
        description="Terraform modularity"
    ),
    TestQuery(
        query="How do I manage Terraform state in a team environment?",
        category="terraform",
        complexity="medium",
        expected_sources=["terraform"],
        description="Terraform state management"
    ),
    TestQuery(
        query="Explain Terraform workspaces and when to use them vs separate state files",
        category="terraform",
        complexity="complex",
        expected_sources=["terraform"],
        description="Advanced Terraform patterns"
    ),

    # Docker (4 queries)
    TestQuery(
        query="How do I write a Dockerfile for a Python application?",
        category="docker",
        complexity="simple",
        expected_sources=["docker"],
        description="Basic Dockerfile creation"
    ),
    TestQuery(
        query="What is the difference between CMD and ENTRYPOINT in Docker?",
        category="docker",
        complexity="medium",
        expected_sources=["docker"],
        description="Dockerfile instructions"
    ),
    TestQuery(
        query="How do I optimize Docker image size for production?",
        category="docker",
        complexity="medium",
        expected_sources=["docker"],
        description="Docker optimization"
    ),
    TestQuery(
        query="Explain Docker multi-stage builds and layer caching strategies",
        category="docker",
        complexity="complex",
        expected_sources=["docker"],
        description="Advanced Docker patterns"
    ),

    # Ansible (2 queries)
    TestQuery(
        query="How do I write an Ansible playbook to install packages?",
        category="ansible",
        complexity="simple",
        expected_sources=["ansible"],
        description="Basic Ansible playbook"
    ),
    TestQuery(
        query="What are Ansible roles and how do I structure them?",
        category="ansible",
        complexity="medium",
        expected_sources=["ansible"],
        description="Ansible organization"
    ),

    # Prometheus/Monitoring (2 queries)
    TestQuery(
        query="How do I configure Prometheus alerting rules?",
        category="prometheus",
        complexity="medium",
        expected_sources=["prometheus"],
        description="Prometheus alerting"
    ),
    TestQuery(
        query="What PromQL functions should I use for rate calculations?",
        category="prometheus",
        complexity="complex",
        expected_sources=["prometheus"],
        description="Advanced PromQL"
    ),

    # Cross-domain (1 query)
    TestQuery(
        query="How do I set up a CI/CD pipeline that deploys to Kubernetes using Terraform?",
        category="cross-domain",
        complexity="complex",
        expected_sources=["kubernetes", "terraform"],
        description="Multi-tool integration"
    ),
]


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class RetrievalResult:
    """Results from a single retrieval operation"""
    query: str
    latency_ms: float
    num_results: int
    scores: List[float]
    sources: List[str]
    source_types: List[str]
    expected_sources: List[str]
    source_match_rate: float  # Percentage of results matching expected sources
    top_score: float
    avg_score: float
    min_score: float


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run"""
    name: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    use_reranking: bool = False
    reranker_model: Optional[str] = None
    description: str = ""


@dataclass
class ConfigBenchmarkResult:
    """Results for a single configuration"""
    config: BenchmarkConfig
    retrieval_results: List[RetrievalResult]

    # Latency statistics (ms)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_mean: float = 0.0
    latency_std: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0

    # Score statistics
    score_p50: float = 0.0
    score_p95: float = 0.0
    score_mean: float = 0.0
    score_std: float = 0.0

    # Quality metrics
    avg_source_match_rate: float = 0.0
    queries_with_results: int = 0
    total_queries: int = 0


@dataclass
class BenchmarkReport:
    """Complete benchmark report"""
    timestamp: str
    total_duration_seconds: float
    configurations: List[ConfigBenchmarkResult]
    test_queries_count: int
    qdrant_collection: str
    qdrant_points_count: int
    embedding_model: str
    embedding_device: str
    system_info: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Benchmarking Engine
# =============================================================================

class RAGBenchmark:
    """RAG system benchmarking engine"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._init_clients()

    def _init_clients(self):
        """Initialize Qdrant client and embeddings"""
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        if self.verbose:
            print(f"Initializing embedding model on {EMBEDDING_DEVICE}...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': EMBEDDING_DEVICE}
        )

        # Warmup embedding model
        _ = self.embeddings.embed_query("warmup query")

        if self.verbose:
            print("Embedding model ready.")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get Qdrant collection statistics"""
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            return {
                "collection_name": COLLECTION_NAME,
                "points_count": info.points_count or 0,
                "vectors_count": info.vectors_count or 0,
            }
        except Exception as e:
            return {"error": str(e), "points_count": 0}

    def search_with_scores(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[List[Any], List[float], float]:
        """
        Perform vector search and return results with scores and latency.

        Returns:
            Tuple of (documents, scores, latency_ms)
        """
        vectorstore = Qdrant(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embeddings=self.embeddings,
        )

        start_time = time.perf_counter()
        results_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        documents = [doc for doc, _ in results_with_scores]
        # Note: Qdrant returns distance, lower is better for cosine
        # Convert to similarity score (1 - distance for cosine)
        scores = [1 - score for _, score in results_with_scores]

        return documents, scores, latency_ms

    def benchmark_single_query(
        self,
        test_query: TestQuery,
        config: BenchmarkConfig
    ) -> RetrievalResult:
        """Benchmark a single query with given configuration"""

        docs, scores, latency_ms = self.search_with_scores(
            test_query.query,
            top_k=config.top_k
        )

        # Extract source information
        sources = [
            doc.metadata.get('source', 'Unknown')
            for doc in docs
        ]
        source_types = [
            doc.metadata.get('source_type', 'Unknown')
            for doc in docs
        ]

        # Calculate source match rate
        matched = sum(
            1 for st in source_types
            if st in test_query.expected_sources
        )
        match_rate = matched / len(source_types) if source_types else 0.0

        return RetrievalResult(
            query=test_query.query,
            latency_ms=latency_ms,
            num_results=len(docs),
            scores=scores,
            sources=sources,
            source_types=source_types,
            expected_sources=test_query.expected_sources,
            source_match_rate=match_rate,
            top_score=max(scores) if scores else 0.0,
            avg_score=statistics.mean(scores) if scores else 0.0,
            min_score=min(scores) if scores else 0.0,
        )

    def benchmark_configuration(
        self,
        config: BenchmarkConfig,
        queries: List[TestQuery],
        warmup_runs: int = 2
    ) -> ConfigBenchmarkResult:
        """Benchmark all queries for a given configuration"""

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {config.name}")
            print(f"  top_k={config.top_k}, reranking={config.use_reranking}")
            print(f"{'='*60}")

        # Warmup runs (not counted)
        if warmup_runs > 0 and self.verbose:
            print(f"Performing {warmup_runs} warmup runs...")
        for _ in range(warmup_runs):
            self.search_with_scores(queries[0].query, top_k=config.top_k)

        # Run benchmarks
        results = []
        iterator = tqdm(queries, desc="Queries") if self.verbose else queries

        for test_query in iterator:
            result = self.benchmark_single_query(test_query, config)
            results.append(result)

        # Calculate statistics
        latencies = [r.latency_ms for r in results]
        all_scores = [s for r in results for s in r.scores]

        config_result = ConfigBenchmarkResult(
            config=config,
            retrieval_results=results,
            latency_p50=float(np.percentile(latencies, 50)),
            latency_p95=float(np.percentile(latencies, 95)),
            latency_p99=float(np.percentile(latencies, 99)),
            latency_mean=float(np.mean(latencies)),
            latency_std=float(np.std(latencies)),
            latency_min=float(np.min(latencies)),
            latency_max=float(np.max(latencies)),
            score_p50=float(np.percentile(all_scores, 50)) if all_scores else 0.0,
            score_p95=float(np.percentile(all_scores, 95)) if all_scores else 0.0,
            score_mean=float(np.mean(all_scores)) if all_scores else 0.0,
            score_std=float(np.std(all_scores)) if all_scores else 0.0,
            avg_source_match_rate=float(np.mean([r.source_match_rate for r in results])),
            queries_with_results=sum(1 for r in results if r.num_results > 0),
            total_queries=len(queries),
        )

        return config_result

    def run_benchmark(
        self,
        configs: List[BenchmarkConfig],
        queries: Optional[List[TestQuery]] = None
    ) -> BenchmarkReport:
        """Run complete benchmark across all configurations"""

        if queries is None:
            queries = TEST_QUERIES

        start_time = time.time()

        # Get collection stats
        stats = self.get_collection_stats()

        if self.verbose:
            print(f"\nRAG Benchmark Starting")
            print(f"Collection: {COLLECTION_NAME}")
            print(f"Points: {stats.get('points_count', 'N/A')}")
            print(f"Test queries: {len(queries)}")
            print(f"Configurations: {len(configs)}")

        # Run benchmarks for each configuration
        config_results = []
        for config in configs:
            result = self.benchmark_configuration(config, queries)
            config_results.append(result)

        total_duration = time.time() - start_time

        # Create report
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            total_duration_seconds=total_duration,
            configurations=config_results,
            test_queries_count=len(queries),
            qdrant_collection=COLLECTION_NAME,
            qdrant_points_count=stats.get('points_count', 0),
            embedding_model=EMBEDDING_MODEL,
            embedding_device=EMBEDDING_DEVICE,
            system_info={
                "qdrant_host": QDRANT_HOST,
                "qdrant_port": QDRANT_PORT,
            }
        )

        return report


# =============================================================================
# Reporting Functions
# =============================================================================

def print_summary_report(report: BenchmarkReport):
    """Print formatted summary report to console"""

    print("\n" + "="*80)
    print("RAG BENCHMARK SUMMARY REPORT")
    print("="*80)

    print(f"\nTimestamp: {report.timestamp}")
    print(f"Total Duration: {report.total_duration_seconds:.2f} seconds")
    print(f"Collection: {report.qdrant_collection} ({report.qdrant_points_count} points)")
    print(f"Embedding Model: {report.embedding_model} ({report.embedding_device})")
    print(f"Test Queries: {report.test_queries_count}")

    # Latency comparison table
    print("\n" + "-"*80)
    print("LATENCY COMPARISON (milliseconds)")
    print("-"*80)

    latency_data = []
    for cr in report.configurations:
        latency_data.append([
            cr.config.name,
            f"{cr.latency_p50:.2f}",
            f"{cr.latency_p95:.2f}",
            f"{cr.latency_p99:.2f}",
            f"{cr.latency_mean:.2f}",
            f"{cr.latency_std:.2f}",
            f"{cr.latency_min:.2f}",
            f"{cr.latency_max:.2f}",
        ])

    print(tabulate(
        latency_data,
        headers=["Config", "P50", "P95", "P99", "Mean", "Std", "Min", "Max"],
        tablefmt="grid"
    ))

    # Score comparison table
    print("\n" + "-"*80)
    print("RETRIEVAL SCORE COMPARISON")
    print("-"*80)

    score_data = []
    for cr in report.configurations:
        score_data.append([
            cr.config.name,
            f"{cr.score_p50:.4f}",
            f"{cr.score_p95:.4f}",
            f"{cr.score_mean:.4f}",
            f"{cr.score_std:.4f}",
            f"{cr.avg_source_match_rate*100:.1f}%",
        ])

    print(tabulate(
        score_data,
        headers=["Config", "Score P50", "Score P95", "Score Mean", "Score Std", "Source Match"],
        tablefmt="grid"
    ))

    # Per-category breakdown
    print("\n" + "-"*80)
    print("PER-CATEGORY LATENCY (P50, ms) - Best Configuration")
    print("-"*80)

    # Use first configuration for category breakdown
    if report.configurations:
        best_config = report.configurations[0]
        category_latencies: Dict[str, List[float]] = {}

        for i, result in enumerate(best_config.retrieval_results):
            query = TEST_QUERIES[i] if i < len(TEST_QUERIES) else None
            if query:
                category = query.category
                if category not in category_latencies:
                    category_latencies[category] = []
                category_latencies[category].append(result.latency_ms)

        category_data = []
        for category, latencies in sorted(category_latencies.items()):
            category_data.append([
                category,
                len(latencies),
                f"{np.percentile(latencies, 50):.2f}",
                f"{np.mean(latencies):.2f}",
            ])

        print(tabulate(
            category_data,
            headers=["Category", "Queries", "P50 Latency", "Mean Latency"],
            tablefmt="grid"
        ))

    # Complexity breakdown
    print("\n" + "-"*80)
    print("PER-COMPLEXITY LATENCY (P50, ms)")
    print("-"*80)

    if report.configurations:
        best_config = report.configurations[0]
        complexity_latencies: Dict[str, List[float]] = {}

        for i, result in enumerate(best_config.retrieval_results):
            query = TEST_QUERIES[i] if i < len(TEST_QUERIES) else None
            if query:
                complexity = query.complexity
                if complexity not in complexity_latencies:
                    complexity_latencies[complexity] = []
                complexity_latencies[complexity].append(result.latency_ms)

        complexity_data = []
        for complexity in ["simple", "medium", "complex"]:
            if complexity in complexity_latencies:
                latencies = complexity_latencies[complexity]
                complexity_data.append([
                    complexity,
                    len(latencies),
                    f"{np.percentile(latencies, 50):.2f}",
                    f"{np.mean(latencies):.2f}",
                ])

        print(tabulate(
            complexity_data,
            headers=["Complexity", "Queries", "P50 Latency", "Mean Latency"],
            tablefmt="grid"
        ))

    print("\n" + "="*80)


def print_detailed_results(report: BenchmarkReport, config_name: Optional[str] = None):
    """Print detailed per-query results"""

    print("\n" + "="*80)
    print("DETAILED QUERY RESULTS")
    print("="*80)

    for cr in report.configurations:
        if config_name and cr.config.name != config_name:
            continue

        print(f"\nConfiguration: {cr.config.name}")
        print("-"*60)

        query_data = []
        for i, result in enumerate(cr.retrieval_results):
            query = TEST_QUERIES[i] if i < len(TEST_QUERIES) else None
            category = query.category if query else "N/A"

            query_data.append([
                result.query[:40] + "..." if len(result.query) > 40 else result.query,
                category,
                f"{result.latency_ms:.2f}",
                result.num_results,
                f"{result.top_score:.4f}",
                f"{result.source_match_rate*100:.0f}%",
            ])

        print(tabulate(
            query_data,
            headers=["Query", "Category", "Latency (ms)", "Results", "Top Score", "Match %"],
            tablefmt="grid"
        ))


def export_results_json(report: BenchmarkReport, output_path: str):
    """Export benchmark results to JSON file"""

    def serialize(obj):
        """Custom serializer for dataclasses and numpy types"""
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Convert to dict manually to handle nested dataclasses
    report_dict = {
        "timestamp": report.timestamp,
        "total_duration_seconds": report.total_duration_seconds,
        "test_queries_count": report.test_queries_count,
        "qdrant_collection": report.qdrant_collection,
        "qdrant_points_count": report.qdrant_points_count,
        "embedding_model": report.embedding_model,
        "embedding_device": report.embedding_device,
        "system_info": report.system_info,
        "configurations": []
    }

    for cr in report.configurations:
        config_dict = {
            "config": asdict(cr.config),
            "latency_p50": cr.latency_p50,
            "latency_p95": cr.latency_p95,
            "latency_p99": cr.latency_p99,
            "latency_mean": cr.latency_mean,
            "latency_std": cr.latency_std,
            "latency_min": cr.latency_min,
            "latency_max": cr.latency_max,
            "score_p50": cr.score_p50,
            "score_p95": cr.score_p95,
            "score_mean": cr.score_mean,
            "score_std": cr.score_std,
            "avg_source_match_rate": cr.avg_source_match_rate,
            "queries_with_results": cr.queries_with_results,
            "total_queries": cr.total_queries,
            "retrieval_results": [asdict(r) for r in cr.retrieval_results]
        }
        report_dict["configurations"].append(config_dict)

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2, default=serialize)

    print(f"\nResults exported to: {output_path}")


def analyze_score_distribution(report: BenchmarkReport):
    """Analyze and print score distribution statistics"""

    print("\n" + "="*80)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("="*80)

    for cr in report.configurations:
        print(f"\nConfiguration: {cr.config.name}")
        print("-"*60)

        all_scores = [s for r in cr.retrieval_results for s in r.scores]

        if not all_scores:
            print("No scores available.")
            continue

        # Basic statistics
        print(f"  Total scores: {len(all_scores)}")
        print(f"  Mean: {np.mean(all_scores):.4f}")
        print(f"  Median: {np.median(all_scores):.4f}")
        print(f"  Std Dev: {np.std(all_scores):.4f}")
        print(f"  Min: {np.min(all_scores):.4f}")
        print(f"  Max: {np.max(all_scores):.4f}")

        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\n  Percentiles:")
        for p in percentiles:
            print(f"    P{p}: {np.percentile(all_scores, p):.4f}")

        # Score buckets
        buckets = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        print(f"\n  Score distribution:")
        for i in range(len(buckets) - 1):
            count = sum(1 for s in all_scores if buckets[i] <= s < buckets[i+1])
            pct = count / len(all_scores) * 100
            print(f"    [{buckets[i]:.1f}, {buckets[i+1]:.1f}): {count} ({pct:.1f}%)")


# =============================================================================
# Main Entry Point
# =============================================================================

def get_default_configurations() -> List[BenchmarkConfig]:
    """Get default benchmark configurations for comparison"""
    return [
        BenchmarkConfig(
            name="baseline_k5",
            top_k=5,
            chunk_size=1000,
            chunk_overlap=200,
            use_reranking=False,
            description="Default configuration with top_k=5"
        ),
        BenchmarkConfig(
            name="expanded_k10",
            top_k=10,
            chunk_size=1000,
            chunk_overlap=200,
            use_reranking=False,
            description="Expanded retrieval with top_k=10"
        ),
        BenchmarkConfig(
            name="minimal_k3",
            top_k=3,
            chunk_size=1000,
            chunk_overlap=200,
            use_reranking=False,
            description="Minimal retrieval with top_k=3"
        ),
        BenchmarkConfig(
            name="large_k20",
            top_k=20,
            chunk_size=1000,
            chunk_overlap=200,
            use_reranking=False,
            description="Large retrieval pool with top_k=20"
        ),
        # Placeholder for reranking configurations (Phase 2)
        # BenchmarkConfig(
        #     name="reranked_k20_to_5",
        #     top_k=20,  # Initial retrieval
        #     chunk_size=1000,
        #     chunk_overlap=200,
        #     use_reranking=True,
        #     reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        #     description="Retrieve 20, rerank to top 5"
        # ),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RAG system retrieval performance"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path (default: benchmark_results_<timestamp>.json)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-query results"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=None,
        help="Custom top_k values to test (e.g., --top-k 3 5 10 20)"
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=None,
        help="Number of queries to run (default: all 20)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter queries by category (e.g., kubernetes, terraform)"
    )

    args = parser.parse_args()

    # Initialize benchmark engine
    benchmark = RAGBenchmark(verbose=not args.quiet)

    # Prepare configurations
    if args.top_k:
        configs = [
            BenchmarkConfig(
                name=f"top_k_{k}",
                top_k=k,
                chunk_size=1000,
                chunk_overlap=200,
                use_reranking=False,
                description=f"Configuration with top_k={k}"
            )
            for k in args.top_k
        ]
    else:
        configs = get_default_configurations()

    # Prepare queries
    queries = TEST_QUERIES

    if args.category:
        queries = [q for q in queries if q.category == args.category]
        if not queries:
            print(f"No queries found for category: {args.category}")
            sys.exit(1)

    if args.queries:
        queries = queries[:args.queries]

    # Run benchmark
    report = benchmark.run_benchmark(configs, queries)

    # Print summary
    print_summary_report(report)

    # Print detailed results if requested
    if args.detailed:
        print_detailed_results(report)

    # Analyze score distribution
    analyze_score_distribution(report)

    # Export results
    output_path = args.output or f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    export_results_json(report, output_path)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
