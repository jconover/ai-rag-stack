# DevOps AI Assistant - Improvement Roadmap

This document contains recommendations from 12 specialized AI agents reviewing the codebase for improvements.

## Critical Fixes (Do First)

| Issue                         | Details                                                                                | File                    | Status |
| ----------------------------- | -------------------------------------------------------------------------------------- | ----------------------- | ------ |
| **Streaming bug**             | `generate_response_stream()` is async but iterates sync generator, blocking event loop | `rag.py:138-212`        | DONE   |
| **Remove `--reload` in prod** | Dev flag in production Dockerfile                                                      | `backend/Dockerfile:28` | DONE   |
| **CORS wildcard**             | `allow_origins=["*"]` is insecure for production                                       | `main.py:32`            | DONE   |

---

## Quick Wins (Low Effort, High Impact)

### 1. System/User Message Separation

**File:** `backend/app/rag.py` lines 85-97

**Current:** Everything crammed into single `user` message.

**Fix:** Use proper role separation:

```python
messages=[
    {'role': 'system', 'content': system_prompt},  # Persona + instructions
    {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {query}"}
]
```

### 2. Embedding Model Warmup

**File:** `backend/app/vectorstore.py`

Add warmup after model initialization:

```python
_ = self.embeddings.embed_query("warmup")  # Force model load at startup
```

### 3. Redis Pipeline Batching

**File:** `backend/app/main.py` lines 57-65

**Current:** 2 Redis calls per save (rpush + expire).

**Fix:**

```python
def save_message(session_id: str, role: str, content: str):
    history_key = f"chat:{session_id}"
    message = json.dumps({"role": role, "content": content})
    pipe = redis_client.pipeline()
    pipe.rpush(history_key, message)
    pipe.expire(history_key, 86400)
    pipe.execute()
```

### 4. GPU for Embeddings

**Files:** `backend/app/vectorstore.py:23`, `scripts/ingest_docs.py:43`

Change `device: 'cpu'` to `'cuda'` for 5-10x speedup.

### 5. Pin Container Versions

**File:** `docker-compose.yml`

Replace `:latest` tags:

```yaml
ollama: ollama/ollama:0.3.0
qdrant: qdrant/qdrant:v1.7.4
redis: redis:7.2-alpine
```

### 6. Ollama Keep-Alive

**File:** `docker-compose.yml`

Add environment variables:

```yaml
environment:
  - OLLAMA_KEEP_ALIVE=24h
  - OLLAMA_NUM_PARALLEL=4
```

### 7. Uvicorn Workers

**File:** `backend/Dockerfile`

For production, use multiple workers:

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## High-Impact Improvements

### RAG Quality Enhancements

#### 1. Add Reranking Layer (Cross-Encoder)

**Impact:** +20-40% precision | **Effort:** Medium

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query: str, docs: List, top_k: int = 5):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
```

#### 2. Implement Hybrid Search (BM25 + Vector) ✅

**Impact:** Better keyword queries | **Effort:** Medium | **Status:** DONE

Implementation uses Qdrant native sparse vectors with fastembed BM25:

```python
# Reciprocal Rank Fusion (RRF) combining dense and sparse results
score = alpha * (1/(k + rank_dense)) + (1-alpha) * (1/(k + rank_sparse))
```

**Files:**
- `backend/app/sparse_encoder.py` - BM25 sparse encoding with fastembed
- `backend/app/vectorstore.py` - `hybrid_search_with_scores()` method
- `backend/app/rag.py` - Integration with RAG pipeline
- `scripts/ingest_docs.py` - Hybrid ingestion support

**Configuration:**
- `HYBRID_SEARCH_ENABLED=true` - Enable hybrid search
- `HYBRID_SEARCH_ALPHA=0.5` - Balance dense/sparse (0.5 = equal weight)
- `HYBRID_RRF_K=60` - RRF constant
- `SPARSE_ENCODER_MODEL=Qdrant/bm25` - Sparse encoder model

**Performance (128k chunks indexed):**
- Retrieval: 15-80ms (semantic + keyword matching)
- Reranking: ~280-350ms
- Total E2E: <500ms before LLM generation

#### 3. Semantic Chunking (Markdown-Aware)

**Impact:** Preserves document structure | **Effort:** Medium

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split = [("#", "h1"), ("##", "h2"), ("###", "h3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split)
```

#### 4. Query Expansion / HyDE ✅

**Impact:** +15-25% recall | **Effort:** Medium | **Status:** DONE

Implementation uses Hypothetical Document Embeddings (HyDE) to generate synthetic
documents for vague queries, improving semantic matching against document chunks.

**Files:**
- `backend/app/query_expansion.py` - HyDEExpander class with DevOps-focused prompts
- `backend/app/config.py` - HyDE configuration settings
- `backend/app/rag.py` - Integration into RAG pipeline

**Features:**
- Smart skip patterns for CLI commands, error messages, file paths
- Configurable timeout and model selection
- Metrics tracking (hyde_used, hyde_time_ms in response)

**Configuration:**
- `HYDE_ENABLED=true` - Enable HyDE query expansion
- `HYDE_MODEL=llama3.1:8b` - Model for generating hypothetical documents
- `HYDE_TEMPERATURE=0.3` - Lower = more focused generation
- `HYDE_MAX_TOKENS=256` - Max length of hypothetical document
- `HYDE_TIMEOUT_SECONDS=10.0` - Timeout for HyDE generation

**Performance:**
- Vague queries: ~2.2s HyDE generation + retrieval
- Specific queries: HyDE skipped (instant, pattern-matched)

#### 5. Web Search Fallback (Tavily) ✅

**Impact:** Unlimited topic coverage | **Effort:** Medium | **Status:** DONE

When local vector search returns low-confidence results, the system falls back to
Tavily web search to find relevant documentation from trusted sources.

**Files:**
- `backend/app/web_search.py` - TavilySearcher class with async/sync support
- `backend/app/config.py` - Web search configuration settings
- `backend/app/rag.py` - Integration into RAG pipeline

**Features:**
- Score-based triggering (configurable threshold, default 0.4)
- Domain whitelist for trusted doc sources (AWS, K8s, Docker, Terraform)
- Results merged into LLM context with source attribution
- Metrics tracking (web_search_used, web_search_time_ms in response)

**Configuration:**
- `WEB_SEARCH_ENABLED=true` - Enable web search fallback
- `TAVILY_API_KEY=tvly-xxx` - Tavily API key (free tier: 1,000/month)
- `WEB_SEARCH_MIN_SCORE_THRESHOLD=0.4` - Trigger when avg score below this
- `WEB_SEARCH_MAX_RESULTS=5` - Number of web results to fetch
- `WEB_SEARCH_INCLUDE_DOMAINS=docs.aws.amazon.com,kubernetes.io` - Trusted domains

**Cost:**
- Free tier: 1,000 searches/month
- Pay-as-you-go: $0.008/search

#### 6. Upgrade Embedding Model

**Impact:** +10-15% retrieval quality | **Effort:** Low

Switch from `all-MiniLM-L6-v2` to `BAAI/bge-base-en-v1.5` (768 dims).

---

### Observability & Analytics

#### 1. Prometheus Metrics Endpoint

Add `/metrics` endpoint tracking:

- Query latency (p50, p95, p99)
- Vector search time
- LLM generation time
- Tokens per second
- Error rates

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

#### 2. Expose Retrieval Similarity Scores

**File:** `backend/app/vectorstore.py`

Switch to `similarity_search_with_score()` to capture and log cosine similarity scores.

#### 3. User Feedback Endpoint

```python
@app.post("/api/feedback")
async def submit_feedback(session_id: str, message_id: str, rating: int, feedback_type: str):
    # Store: query, response, sources, model, rating
    pass
```

#### 4. Query Analytics Pipeline

Persist anonymized query logs with:

- Timestamp, query, model, latency_ms
- Token count, sources returned, retrieval scores

#### 5. LLM-as-Judge Sampling

Sample 5-10% of responses for automated evaluation:

- Groundedness (does response match sources?)
- Completeness (does it fully answer?)
- Hallucination detection

---

### Infrastructure Improvements

#### 1. GitHub Actions CI/CD ✅

**Status:** DONE

**Files Created:**
- `.github/workflows/ci.yml` - Main CI pipeline (lint, typecheck, test, build)
- `.github/workflows/docker-publish.yml` - Multi-platform Docker builds + Docker Hub push
- `.github/workflows/security.yml` - Trivy, pip-audit, Hadolint, Gitleaks scanning
- `.github/dependabot.yml` - Automated dependency updates (Python, npm, Docker, Actions)

**Features:**
- Automated Docker image builds (amd64 + arm64)
- Image vulnerability scanning (Trivy)
- Push to Docker Hub on tagged releases
- Run tests before deployment
- Weekly security scans
- Grouped dependency updates

#### 2. Redis Connection Pooling

**File:** `backend/app/main.py`

```python
redis_pool = redis.ConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    max_connections=50,
    decode_responses=True
)
redis_client = redis.Redis(connection_pool=redis_pool)
```

#### 3. Embedding Cache in Redis

Cache query embeddings to reduce latency 30-50%:

```python
def search(self, query: str, top_k: int = None) -> List[Document]:
    cache_key = f"emb:{hashlib.md5(query.encode()).hexdigest()}"
    cached = redis_client.get(cache_key)
    if cached:
        query_embedding = json.loads(cached)
    else:
        query_embedding = self.embeddings.embed_query(query)
        redis_client.setex(cache_key, 3600, json.dumps(query_embedding))
```

#### 4. Incremental Document Ingestion ✅

**Impact:** Faster updates, reduced redundancy | **Effort:** Medium | **Status:** DONE

Implementation uses SHA-256 content hashing with SQLite registry for change detection.

**Files:**
- `scripts/ingestion_registry.py` - SQLite registry for tracking ingested files
- `scripts/ingest_docs.py` - Updated with incremental logic and CLI flags
- `backend/app/vectorstore.py` - Added `delete_by_source()` for chunk management

**Features:**
- SHA-256 content hash per file for change detection
- SQLite registry at `data/ingestion_registry.db`
- Automatic detection of new, changed, deleted files
- Chunking config change detection (triggers re-ingestion warning)
- CLI flags for flexible operation

**CLI Usage:**
```bash
python ingest_docs.py              # Incremental (default)
python ingest_docs.py --full       # Force full re-ingestion
python ingest_docs.py --dry-run    # Preview changes without ingesting
python ingest_docs.py --stats      # Show registry statistics
python ingest_docs.py --clear-registry  # Reset registry
```

**Performance:**
- First run: Full ingestion (same as before)
- Subsequent runs: Only new/changed files processed
- Change detection: ~5-10s for 10k files (hash computation)

#### 5. Add PostgreSQL

For structured data:

- Document registry (ingestion metadata)
- Query analytics
- User feedback storage
- Audit trails

---

### Database Optimizations

#### 1. Qdrant Payload Indexing

```python
from qdrant_client.models import PayloadSchemaType

self.client.create_payload_index(
    collection_name=self.collection_name,
    field_name="source_type",
    field_schema=PayloadSchemaType.KEYWORD
)
```

#### 2. Single Vectorstore Instance

**File:** `backend/app/vectorstore.py`

Initialize LangChain `Qdrant()` wrapper once in `__init__`, not per query.

#### 3. HNSW Index Tuning

```python
hnsw_config=HnswConfigDiff(
    m=32,              # Connections per node (default 16)
    ef_construct=200,  # Build-time accuracy (default 100)
)
```

---

## Prioritized Roadmap

### Phase 1: Stability & Performance ✅

- [x] Fix streaming async bug
- [x] Remove `--reload`, add resource limits, add Uvicorn workers
- [x] Add embedding warmup + Ollama keep-alive
- [x] Configurable GPU embeddings (EMBEDDING_DEVICE env var)
- [x] Pin container versions (ollama:0.5.4, qdrant:v1.12.4, redis:7.4-alpine)
- [x] Redis pipeline batching
- [x] Configurable CORS origins

### Phase 2: RAG Quality ✅

- [x] System/user message separation
- [x] Add cross-encoder reranking (reranker.py with ms-marco-MiniLM-L-6-v2)
- [x] Implement semantic chunking (chunkers.py with markdown-aware splitting)
- [x] Add retrieval score logging (metrics.py with JSON + Prometheus)
- [x] Vector store optimizations (HNSW tuning, INT8 quantization)
- [x] Benchmark framework (scripts/benchmark_rag.py)
- [x] Reranker tested and verified working (rerank scores in API response)
- [x] Qdrant client compatibility fix (pinned to 1.11-1.13.x for server v1.12.4)

### Phase 3: Observability

- [x] Prometheus metrics endpoint (optional via ENABLE_PROMETHEUS_METRICS)
- [x] Query analytics logging (retrieval_metrics.jsonl with /tmp fallback)
- [x] Feedback collection endpoint (`/api/feedback`, `/api/feedback/summary`)
- [x] Grafana dashboard (3 dashboards: RAG Performance, User Feedback, System Health)

### Phase 4: Advanced Features (Month 2)

- [x] Hybrid search (BM25 + vector) with RRF fusion - 15-80ms retrieval on 128k docs
- [x] HyDE query expansion for vague queries (~2.2s generation, smart skip patterns)
- [x] Web search fallback (Tavily) for queries with low local retrieval scores
- [x] Incremental ingestion with change detection (SHA-256 hash tracking, SQLite registry)
- [x] GitHub Actions CI/CD (ci.yml, docker-publish.yml, security.yml, dependabot.yml)
- [ ] PostgreSQL for analytics/metadata
- [ ] A/B testing framework
- [ ] User accounts & persistent sessions

---

## Features (Future)

1. **Semantic caching** - Cache responses for semantically similar queries
2. **Auto-expanding documentation** - Detect query patterns with no good results → flag as content gaps
3. **Model comparison mode** - A/B test different LLMs on same query
4. ~~**Query rewriting**~~ - ✅ Implemented via HyDE query expansion
5. **Source freshness tracking** - Show "last updated X days ago" for each source

---

## Prompt Engineering Improvements

### Output Format Specifications

Add to system prompt:

```
Output Format:
- Start with a direct answer (1-2 sentences)
- Provide code examples in fenced code blocks with language tags
- Use bullet points for multi-step procedures
- End with "Sources used: [1], [2]..." when referencing context
- If no relevant context, state: "Based on general knowledge (no documentation matched):"
```

### Parameterized Templates

Convert static templates to have input variables:

````python
{
    "id": "explain-error",
    "prompt": "I'm getting this error:\n```\n{{error_message}}\n```\nContext: {{context}}",
    "variables": [
        {"name": "error_message", "label": "Error Message", "type": "textarea", "required": True},
        {"name": "context", "label": "What were you doing?", "type": "text", "required": False}
    ]
}
````

### Chain-of-Thought for Debugging

For debugging/troubleshooting templates, inject CoT:

```
Think through this step-by-step:
1. First, identify what the error/symptom indicates
2. List possible root causes (most likely first)
3. For each cause, provide a diagnostic command
4. Suggest fixes in order of likelihood
```

---

## Data Pipeline Improvements

### Source Manifest (Single Source of Truth)

Create `sources.yaml`:

```yaml
sources:
  - name: kubernetes
    repo: https://github.com/kubernetes/website.git
    docs_path: content/en/docs
    enabled: true
```

Both `download_docs.sh` and `ingest_docs.py` should read from this manifest.

### Data Quality Validation

Pre-ingestion checks:

- Minimum content length
- Valid markdown structure
- Language detection

Post-ingestion checks:

- Embedding vector validation
- Duplicate detection using similarity threshold

### Batch Processing with Checkpointing

- Process documents in batches (500 chunks per batch)
- Checkpoint progress to Redis
- Enable resume from last successful batch on failure

---

_Document generated: 2025-01-04_
_Last updated: 2026-01-06_
_Based on reviews from 12 specialized AI agents_
