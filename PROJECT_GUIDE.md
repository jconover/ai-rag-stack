# DevOps AI Assistant - Complete Project Guide

**A plain-English explanation of how this RAG system works, written for someone new to AI/ML.**

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [System Architecture Overview](#system-architecture-overview)
3. [The RAG Pipeline Explained](#the-rag-pipeline-explained)
4. [Embeddings and Vector Search](#embeddings-and-vector-search)
5. [Document Ingestion](#document-ingestion)
6. [Advanced RAG Features](#advanced-rag-features)
7. [Data Storage Systems](#data-storage-systems)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Measuring RAG Quality](#measuring-rag-quality)
10. [Quick Reference](#quick-reference)

---

## What This Project Does

The DevOps AI Assistant is a chat application that answers questions about DevOps topics (Kubernetes, Docker, Terraform, AWS, etc.).

**The key difference from ChatGPT:** Instead of relying solely on what the AI learned during training, this system **searches through 30+ documentation sources first**, finds relevant passages, then uses those passages as context when generating answers.

This technique is called **RAG (Retrieval-Augmented Generation)**:
- **Retrieval**: Find relevant documents
- **Augmented**: Add them to the AI's context
- **Generation**: AI generates answer using that context

**Why RAG matters:**
- Reduces hallucinations (AI making things up)
- Provides source citations
- Stays current with documentation updates
- Works with your specific documentation

---

## System Architecture Overview

### The 8 Services

```
┌─────────────────────────────────────────────────────────────────┐
│                        User's Browser                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  FRONTEND (React, port 3000)                                    │
│  The web interface users interact with                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  BACKEND (FastAPI/Python, port 8000)                            │
│  The "brain" - orchestrates everything                          │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Ollama  │  │  Qdrant  │  │  Redis   │  │ Postgres │        │
│  │  :11434  │  │  :6333   │  │  :6379   │  │  :5432   │        │
│  │          │  │          │  │          │  │          │        │
│  │ Runs AI  │  │ Vector   │  │ Cache &  │  │Analytics │        │
│  │ models   │  │ search   │  │ sessions │  │ & logs   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌──────────────────┐    ┌──────────────────┐
        │  Prometheus      │    │  Grafana         │
        │  :9090           │    │  :3001           │
        │  Collects metrics│    │  Dashboards      │
        └──────────────────┘    └──────────────────┘
```

### What Each Service Does

| Service | Port | Purpose |
|---------|------|---------|
| **Frontend** | 3000 | React web app - what users see |
| **Backend** | 8000 | Python API - handles all logic |
| **Ollama** | 11434 | Runs LLMs locally (llama3.1, mistral) |
| **Qdrant** | 6333 | Vector database for semantic search |
| **Redis** | 6379 | Fast cache for sessions & embeddings |
| **PostgreSQL** | 5432 | Stores analytics, users, experiments |
| **Prometheus** | 9090 | Collects performance metrics |
| **Grafana** | 3001 | Visualizes metrics as dashboards |

### Why Docker Compose?

All services run in **containers** - isolated environments that include everything needed to run. Docker Compose orchestrates all 8 containers with a single command:

```bash
make start   # Starts everything
make stop    # Stops everything
```

---

## The RAG Pipeline Explained

When you ask "How do I create a Kubernetes deployment?", here's what happens:

### Step-by-Step Flow

```
1. YOUR QUESTION ARRIVES
   "How do I create a Kubernetes deployment?"
                    │
                    ▼
2. QUERY EXPANSION (Optional - HyDE)
   Short question → Generate hypothetical answer
   This helps match against longer documentation
                    │
                    ▼
3. DOCUMENT RETRIEVAL
   ┌─────────────────────────────────────┐
   │ Vector Search (semantic meaning)    │
   │ + BM25 Search (exact keywords)      │
   │ = Hybrid Results (best of both)     │
   └─────────────────────────────────────┘
   Returns top 20 candidate documents
                    │
                    ▼
4. RERANKING
   Cross-encoder model re-scores candidates
   Top 20 → Best 5 (more accurate selection)
                    │
                    ▼
5. CONTEXT BUILDING
   Format the 5 documents into a prompt:
   "[Source 1 - kubernetes] ..."
   "[Source 2 - kubernetes] ..."
                    │
                    ▼
6. LLM GENERATION
   System prompt + Context + Your question
   → Ollama (llama3.1) → Answer with citations
                    │
                    ▼
7. RESPONSE
   "To create a Kubernetes deployment, use kubectl apply..."
   Sources: [kubernetes.io/docs/...]
```

### Why Not Just Ask the LLM Directly?

| Without RAG | With RAG |
|-------------|----------|
| Knowledge frozen at training time | Access to current documentation |
| May confidently make up answers | Answers grounded in real docs |
| No source citations | Can cite exact sources |
| Generic knowledge only | Your specific documentation |

---

## Embeddings and Vector Search

### What Are Embeddings?

Embeddings convert text into numbers (vectors) that capture **meaning**. Similar concepts get similar numbers.

**Analogy: GPS coordinates for ideas**

```
"Kubernetes pod"     → [0.23, -0.45, 0.67, ...]  (768 numbers)
"K8s container unit" → [0.21, -0.44, 0.68, ...]  (very similar!)
"Pizza recipe"       → [-0.89, 0.12, -0.33, ...] (very different)
```

The model used is `BAAI/bge-base-en-v1.5` - it understands technical language well and outputs 768-dimensional vectors.

### How Vector Search Works

1. **Your question** gets converted to a 768-number vector
2. **Qdrant** (vector database) has pre-computed vectors for all documentation
3. **Similarity search** finds documents with vectors closest to yours
4. **Cosine similarity** measures the "angle" between vectors:
   - 1.0 = identical meaning
   - 0.7+ = very similar (good match)
   - 0.3 = weakly related
   - 0.0 = unrelated

### The Embedding Cache (Redis)

Computing embeddings takes ~50ms. To speed up repeated queries:

```
First ask: "How to create a pod?"
  → Compute embedding (50ms)
  → Store in Redis cache
  → Search Qdrant

Second ask: "How to create a pod?"
  → Check Redis cache (1ms) ✓ Found!
  → Skip computation
  → Search Qdrant
```

Result: **30-50% faster** for common queries.

---

## Document Ingestion

### Where Documents Come From

The `scripts/download_docs.sh` script clones 30+ GitHub repositories:

- **Kubernetes**, Docker, Terraform, Ansible, Helm
- **AWS**, Azure, GCP documentation
- **Prometheus**, Grafana, Elasticsearch
- **Python**, Go, Rust, JavaScript guides

### The Chunking Process

Large documents must be split into smaller pieces because:
1. LLMs have limited context windows
2. Smaller chunks = more precise retrieval

**How it works:**

```
Big Kubernetes Guide (50 pages)
            │
            ▼
    Split by headings (h1, h2, h3)
            │
            ▼
    ┌─────────────────┐
    │ Chunk 1: Pods   │  ~1000 characters each
    │ Chunk 2: Services│  with 200 char overlap
    │ Chunk 3: Deploy │
    │ ...             │
    └─────────────────┘
            │
            ▼
    Convert each to embedding (768 numbers)
            │
            ▼
    Store in Qdrant with metadata
    (source file, heading path, etc.)
```

### Smart Chunking Features

- **Preserves code blocks** - never splits in the middle of code
- **Respects markdown structure** - splits at headings
- **Content-type aware** - different sizes for prose vs. code vs. tables
- **Overlap** - chunks share 200 characters to maintain context

### Incremental Updates

The **ingestion registry** (SQLite database) tracks what's been processed:

```
File: /docs/kubernetes/pods.md
Hash: a3f2b7c9...  (fingerprint of content)
Chunks: 15
Ingested: 2024-01-15
```

When you run `make ingest`:
1. Compare current files to registry
2. Only process **new or changed** files
3. Skip unchanged files (huge time savings)

---

## Advanced RAG Features

This isn't a basic RAG system. It includes four advanced techniques:

### 1. HyDE (Hypothetical Document Embeddings)

**Problem:** Short queries don't match well against long documents.

**Solution:** Generate a hypothetical answer first, then search with that.

```
User: "kubernetes networking"
           │
           ▼ (LLM generates)
"Kubernetes networking enables pods to communicate using CNI plugins.
 Each pod receives a unique IP address. Services provide stable endpoints..."
           │
           ▼ (search with this instead)
Much better matches!
```

**Skipped for:** Error messages, CLI commands, file paths (already specific enough)

### 2. Hybrid Search (Semantic + Keyword)

**Problem:** Pure semantic search can miss exact terms like "kubectl" or "ETCDCTL_API".

**Solution:** Combine both approaches:

| Method | Strength |
|--------|----------|
| **Semantic (dense)** | Understands meaning: "car" = "automobile" |
| **Keyword (BM25)** | Exact matching: "kubectl" = "kubectl" |

Results are merged using **Reciprocal Rank Fusion (RRF)** - a smart algorithm that combines ranked lists.

### 3. Cross-Encoder Reranking

**Problem:** Fast vector search can have false positives.

**Solution:** Two-stage retrieval:

```
Stage 1: Vector search (fast, gets 20 candidates)
           │
           ▼
Stage 2: Cross-encoder (slow but accurate, picks best 5)
```

The cross-encoder sees query AND document together, so it understands relevance much better.

### 4. Web Search Fallback

**Problem:** Local docs might not cover everything.

**Solution:** When local results have low scores, search the web:

```
Local search scores too low (< 0.4)?
           │
           ▼
Search trusted documentation sites via Tavily API
(kubernetes.io, docs.aws.amazon.com, etc.)
           │
           ▼
Add web results to context
```

---

## Data Storage Systems

### Why Three Databases?

Each database is optimized for different tasks:

| Database | Type | Best For |
|----------|------|----------|
| **PostgreSQL** | Relational | Structured records, complex queries, analytics |
| **Qdrant** | Vector | Similarity search, semantic matching |
| **Redis** | Key-Value | Fast cache, session storage, temporary data |

**Analogy - A Kitchen:**
- **PostgreSQL** = Filing cabinet (permanent records)
- **Qdrant** = Recipe index (find similar things)
- **Redis** = Countertop (quick access to current items)

### What's Stored Where

**PostgreSQL** (permanent, structured):
- User accounts and sessions
- Query logs (every question asked)
- Feedback (thumbs up/down)
- A/B test experiments and results
- Analytics aggregations

**Qdrant** (vectors + content):
- Document chunks (the actual text)
- Embedding vectors (768 numbers per chunk)
- Metadata (source file, category, etc.)

**Redis** (temporary, fast):
- Conversation history (24-hour TTL)
- Embedding cache (1-hour TTL)
- Session data

### Data Flow Example

```
User sends message
       │
       ▼
┌─────────────┐
│   REDIS     │ ← Get conversation history
└─────────────┘
       │
       ▼
┌─────────────┐
│   REDIS     │ ← Check embedding cache
└─────────────┘
       │
       ▼ (cache miss)
┌─────────────┐
│   QDRANT    │ ← Vector search for documents
└─────────────┘
       │
       ▼
┌─────────────┐
│   OLLAMA    │ ← Generate response
└─────────────┘
       │
       ▼
┌─────────────┐
│   REDIS     │ ← Save to conversation history
└─────────────┘
       │
       ▼ (async, non-blocking)
┌─────────────┐
│ POSTGRESQL  │ ← Log query for analytics
└─────────────┘
```

---

## Monitoring and Observability

### Why Monitor AI Systems?

AI systems can fail **silently** - returning confident but wrong answers. Traditional software crashes; AI systems degrade. Monitoring helps catch problems early.

### Key Metrics Tracked

| Metric | What It Measures | Healthy Range |
|--------|------------------|---------------|
| **Retrieval latency** | Time to find documents | < 100ms |
| **Total latency** | End-to-end response time | < 2000ms |
| **Retrieval score** | How well docs match query | > 0.7 |
| **Helpful rate** | User thumbs up percentage | > 70% |
| **Cache hit rate** | Embedding cache efficiency | > 30% |

### The Monitoring Stack

**Prometheus** collects metrics every 15 seconds:
```yaml
scrape_configs:
  - job_name: 'rag-backend'
    targets: ['backend:8000']
    metrics_path: '/metrics/'
```

**Grafana** displays three pre-built dashboards:
1. **RAG Performance** - Latency, scores, query rates
2. **User Feedback** - Helpful rate, feedback trends
3. **System Health** - Service status, error rates

### A/B Testing

Compare different configurations scientifically:

```
Experiment: "Model Comparison"
├── Variant A (50%): llama3.1:8b
└── Variant B (50%): mistral:7b

After 1000 queries:
- Variant A: avg latency 1200ms, helpful rate 72%
- Variant B: avg latency 900ms, helpful rate 68%

Statistical analysis determines if difference is significant.
```

---

## Measuring RAG Quality

### The Core Metrics

**MRR (Mean Reciprocal Rank)** - "How quickly do I find the right document?"

| First relevant doc at position | MRR Score |
|-------------------------------|-----------|
| 1st | 1.0 (perfect) |
| 2nd | 0.5 |
| 3rd | 0.33 |
| 5th | 0.2 |
| Not found | 0.0 |

**Recall** - "Did I find ALL relevant documents?"
```
3 relevant docs exist, found 2 → Recall = 2/3 = 0.67
```

**Precision** - "Of what I found, how much is actually relevant?"
```
Retrieved 5 docs, 4 are relevant → Precision = 4/5 = 0.80
```

### Quality Benchmarks

| Metric | Bad | OK | Good | Excellent |
|--------|-----|-----|------|-----------|
| MRR@5 | <0.3 | 0.3-0.5 | 0.5-0.7 | >0.7 |
| Recall@5 | <0.4 | 0.4-0.6 | 0.6-0.8 | >0.8 |
| Precision@5 | <0.4 | 0.4-0.6 | 0.6-0.8 | >0.8 |

### The Evaluation System

The project includes 16 test questions with known correct answers:

```python
EvalExample(
    query="How do I create a Kubernetes deployment?",
    relevant_doc_ids=["kubernetes", "deployment"],
    expected_keywords=["kubectl apply", "replicas", "spec"],
    category="kubernetes",
)
```

Run evaluation:
```python
from app.evaluation import run_evaluation_suite, format_eval_report
results = await run_evaluation_suite()
print(format_eval_report(results))
```

---

## Quick Reference

### Common Commands

```bash
# Start/Stop
make start          # Start all services
make stop           # Stop all services
make logs-backend   # View backend logs

# Setup
make pull-model     # Download default LLM
make ingest         # Index documentation

# Monitoring
make health         # Check service status
make stats          # Vector database stats
make grafana        # Open Grafana dashboards

# Maintenance
make backup         # Backup all databases
make backup-verify  # Verify backup integrity
```

### Key Configuration (.env)

```bash
# LLM Settings
OLLAMA_DEFAULT_MODEL=llama3.1:8b

# RAG Tuning
TOP_K_RESULTS=5           # Documents per query
CHUNK_SIZE=1000           # Characters per chunk

# Advanced Features
HYBRID_SEARCH_ENABLED=true    # Semantic + keyword
RERANKER_ENABLED=true         # Cross-encoder reranking
HYDE_ENABLED=true             # Query expansion

# Performance
EMBEDDING_DEVICE=cuda         # GPU acceleration
EMBEDDING_CACHE_TTL=3600      # Cache for 1 hour
```

### Key Files

| File | Purpose |
|------|---------|
| `backend/app/rag.py` | Main RAG pipeline |
| `backend/app/vectorstore.py` | Qdrant interface, embeddings |
| `backend/app/reranker.py` | Cross-encoder reranking |
| `backend/app/query_expansion.py` | HyDE implementation |
| `backend/app/evaluation.py` | Quality metrics |
| `scripts/ingest_docs.py` | Document ingestion |
| `docker-compose.yml` | Service definitions |

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat` | POST | Send question, get answer |
| `/api/chat/stream` | POST | Streaming response |
| `/api/health` | GET | Service status |
| `/api/models` | GET | Available LLMs |
| `/api/stats` | GET | Vector DB statistics |
| `/api/feedback` | POST | Submit thumbs up/down |

---

## Glossary

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation - search docs before generating answers |
| **Embedding** | Text converted to numbers (vector) that capture meaning |
| **Vector Database** | Database optimized for similarity search (Qdrant) |
| **LLM** | Large Language Model - AI that generates text (llama3.1) |
| **Chunking** | Splitting documents into smaller searchable pieces |
| **HyDE** | Generate hypothetical answer to improve search |
| **Cross-encoder** | Model that scores query+document together |
| **BM25** | Classic keyword-matching algorithm |
| **RRF** | Reciprocal Rank Fusion - combines multiple result lists |
| **MRR** | Mean Reciprocal Rank - measures ranking quality |

---

*Generated by 8 expert AI agents analyzing the codebase - January 2026*
