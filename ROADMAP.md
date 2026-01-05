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

#### 2. Implement Hybrid Search (BM25 + Vector)

**Impact:** Better keyword queries | **Effort:** Medium

Qdrant supports sparse vectors for hybrid search:

```python
# Combine with reciprocal rank fusion
score = 1/(k + rank_dense) + 1/(k + rank_sparse)
```

#### 3. Semantic Chunking (Markdown-Aware)

**Impact:** Preserves document structure | **Effort:** Medium

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split = [("#", "h1"), ("##", "h2"), ("###", "h3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split)
```

#### 4. Query Expansion / HyDE

**Impact:** +15-25% recall | **Effort:** Medium

```python
def _expand_query(self, query: str, model: str) -> str:
    """Generate hypothetical answer to improve retrieval"""
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': f"Write a brief technical answer to: {query}"}],
        options={'temperature': 0.3, 'num_predict': 150}
    )
    return f"{query}\n{response['message']['content']}"
```

#### 5. Upgrade Embedding Model

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

#### 1. GitHub Actions CI/CD

Create `.github/workflows/ci.yml`:

- Automated Docker image builds
- Image vulnerability scanning (Trivy)
- Push to Docker Hub on tagged releases
- Run tests before deployment

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

#### 4. Incremental Document Ingestion

Add file-level change tracking:

- Hash content + path before embedding
- Query Qdrant for existing hashes
- Skip already-indexed documents
- Track: timestamp, version, source commit SHA

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
- [ ] Grafana dashboard

### Phase 4: Advanced Features (Month 2)

- [ ] Hybrid search (BM25 + vector)
- [ ] PostgreSQL for analytics/metadata
- [ ] Incremental ingestion with change detection
- [ ] A/B testing framework
- [ ] User accounts & persistent sessions

---

## Features (Future)

1. **Semantic caching** - Cache responses for semantically similar queries
2. **Auto-expanding documentation** - Detect query patterns with no good results → flag as content gaps
3. **Model comparison mode** - A/B test different LLMs on same query
4. **Query rewriting** - Use LLM to reformulate vague questions before retrieval
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
_Last updated: 2026-01-04_
_Based on reviews from 12 specialized AI agents_
