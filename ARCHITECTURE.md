# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                     (React Web App)                             │
│                     Port: 3000                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend API                                │
│                    (FastAPI/Python)                             │
│                      Port: 8000                                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              RAG Pipeline                                │  │
│  │  1. Query Processing                                     │  │
│  │  2. Vector Search (Qdrant)                              │  │
│  │  3. Context Formatting                                   │  │
│  │  4. Prompt Building                                      │  │
│  │  5. LLM Generation (Ollama)                             │  │
│  └─────────────────────────────────────────────────────────┘  │
└───┬────────────────┬────────────────┬───────────────────────────┘
    │                │                │
    ▼                ▼                ▼
┌──────────┐  ┌──────────┐  ┌─────────────┐
│  Ollama  │  │  Qdrant  │  │    Redis    │
│   LLM    │  │  Vector  │  │   Memory    │
│  Engine  │  │   Store  │  │   Cache     │
│          │  │          │  │             │
│  11434   │  │   6333   │  │    6379     │
└──────────┘  └──────────┘  └─────────────┘
     │
     ▼
┌──────────┐
│ GPU/CPU  │
│ RTX 3090 │
│  24GB    │
└──────────┘
```

## Data Flow

### 1. Document Ingestion Pipeline

```
Documentation Sources
    │
    ├─ Kubernetes Docs (markdown)
    ├─ Terraform Docs (markdown)
    ├─ Docker Docs (markdown)
    ├─ Ansible Docs (markdown)
    └─ Custom Docs
         │
         ▼
    Document Loader
    (LangChain)
         │
         ▼
    Text Splitter
    (Chunking: 1000 tokens, 200 overlap)
         │
         ▼
    Embedding Model
    (sentence-transformers/all-MiniLM-L6-v2)
    Output: 384-dimensional vectors
         │
         ▼
    Qdrant Vector Store
    (Persistent storage)
```

### 2. Query Processing Flow

```
User Query
    │
    ▼
Backend API (/api/chat)
    │
    ├─ Store in Redis (conversation history)
    │
    ▼
RAG Pipeline
    │
    ├─ 1. Embed query (same model as documents)
    │      Output: 384-dimensional vector
    │
    ├─ 2. Search Qdrant
    │      Cosine similarity search
    │      Top K=5 most relevant chunks
    │
    ├─ 3. Format context
    │      Combine retrieved documents
    │      Add metadata (source, type)
    │
    ├─ 4. Build prompt
    │      System instructions
    │      Context from docs
    │      User query
    │
    └─ 5. Generate with Ollama
         Temperature: 0.7
         Max tokens: 2048
         Stream: No
              │
              ▼
         Response
              │
              ├─ Save to Redis
              │
              └─ Return to user
```

## Component Details

### Ollama (LLM Engine)

**Purpose**: Run large language models locally with GPU acceleration

**Features**:
- GPU acceleration (CUDA)
- Model management
- Multiple concurrent models
- Efficient inference

**Configuration**:
```yaml
Environment:
  OLLAMA_NUM_GPU: 1
  OLLAMA_NUM_THREAD: 16
  OLLAMA_MAX_LOADED_MODELS: 2
  
Resources:
  GPU: NVIDIA RTX 3090 (24GB)
  CPU: 16 threads
```

**Supported Models**:
- Llama 3.1 (8B, 70B)
- CodeLlama (7B, 13B, 34B)
- Mistral (7B)
- DeepSeek Coder (1B, 6B, 33B)

### Qdrant (Vector Database)

**Purpose**: Store and search document embeddings

**Features**:
- Efficient similarity search
- Persistent storage
- REST & gRPC APIs
- Filtering and metadata

**Configuration**:
```yaml
Collection: devops_docs
Vector Size: 384 dimensions
Distance Metric: Cosine similarity
Storage: Persistent volume
```

**Index Structure**:
```
Point {
  id: UUID
  vector: [384 floats]
  payload: {
    text: "chunk content",
    source: "path/to/file.md",
    source_type: "kubernetes",
    metadata: {...}
  }
}
```

### Redis (Memory & Cache)

**Purpose**: Store conversation history and cache

**Features**:
- Fast in-memory storage
- Conversation sessions
- TTL-based expiry
- Pub/sub for future features

**Data Structure**:
```
Key: chat:{session_id}
Type: List
Value: [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
TTL: 24 hours
```

### FastAPI Backend

**Purpose**: REST API and RAG orchestration

**Endpoints**:
```
GET  /                    - API info
GET  /api/health          - Health check
POST /api/chat            - Chat with AI
GET  /api/models          - List models
GET  /api/stats           - Vector DB stats
GET  /api/history/{id}    - Get conversation
```

**Key Classes**:
- `RAGPipeline`: Query processing and generation
- `VectorStore`: Qdrant interface
- `ChatRequest/Response`: API models

### React Frontend

**Purpose**: User interface for chat interaction

**Features**:
- Real-time chat
- Model selection
- Source attribution
- Markdown rendering
- Code syntax highlighting

**Components**:
- `App.js`: Main chat interface
- `MessageList`: Display messages
- `InputForm`: User input
- `ModelSelector`: Choose LLM

## Embedding Model

**Model**: sentence-transformers/all-MiniLM-L6-v2

**Specifications**:
- Dimensions: 384
- Max sequence length: 256 tokens
- Performance: ~3100 sentences/sec on CPU
- Size: ~80MB

**Why this model?**:
- Fast inference on CPU
- Good balance of speed/quality
- Small memory footprint
- Well-suited for semantic search

**Alternatives**:
- `all-mpnet-base-v2` (768 dim, better quality, slower)
- `bge-small-en-v1.5` (384 dim, similar performance)

## RAG Strategy

### Chunking Strategy

```python
Chunk Size: 1000 characters
Overlap: 200 characters
Separator: ["\n\n", "\n", " ", ""]
```

**Rationale**:
- 1000 chars ≈ 250 tokens
- Preserves context across chunks
- Fits well within LLM context window (4096 tokens)

### Retrieval Strategy

```python
Method: Similarity Search
Top K: 5 documents
Distance: Cosine similarity
```

**Context Window Calculation**:
```
User Query: ~100 tokens
Retrieved Docs: 5 × 250 = 1250 tokens
System Prompt: ~150 tokens
Total Input: ~1500 tokens
Response Budget: 2048 tokens
Total: ~3500 tokens (< 4096 limit)
```

### Prompt Template

```
System Instructions (150 tokens)
    │
    ▼
Context from Documentation (1250 tokens)
    [Source 1 - Kubernetes]
    <content>
    ---
    [Source 2 - Docker]
    <content>
    ...
    │
    ▼
User Question (100 tokens)
    │
    ▼
Instructions
    - Answer based on context
    - Provide code examples
    - Be concise
```

## Performance Characteristics

### Latency Breakdown

```
Component               Time (ms)    Percentage
─────────────────────────────────────────────
Query Embedding         50-100       2-3%
Vector Search           100-200      3-5%
Context Formatting      10-20        <1%
LLM Generation          2000-5000    90-95%
Total                   2160-5320    100%
```

**Bottleneck**: LLM generation (GPU-bound)

### Throughput

**Sequential**:
- Llama 3.1 8B: ~40 tokens/sec
- CodeLlama 13B: ~25 tokens/sec
- Mistral 7B: ~50 tokens/sec

**Concurrent** (with 24GB VRAM):
- 2× Llama 8B: ~30 tokens/sec each
- 1× Llama 8B + 1× Mistral 7B: ~35 tokens/sec each

### Resource Usage

**Idle**:
```
CPU: 1-2%
RAM: 2GB (system) + 1GB (embeddings)
GPU: 0.5GB (Ollama idle)
Disk: ~15GB (models) + 5GB (vectors)
```

**Under Load** (Llama 8B):
```
CPU: 10-20% (data prep)
RAM: 4GB
GPU: 5-6GB VRAM
GPU Compute: 70-90%
```

## Scaling Considerations

### Horizontal Scaling

**Current**: Single-node deployment

**Future Options**:
1. **API Layer**: Load balance multiple backend instances
2. **Qdrant**: Distributed cluster mode
3. **Ollama**: Multiple GPU nodes

### Vertical Scaling

**Current Limits** (RTX 3090):
- Max model size: ~20GB (70B quantized)
- Concurrent models: 2-3 small models

**Upgrade Path**:
- 2× RTX 3090 = Run 70B + 8B simultaneously
- A100 (40GB) = Run larger models, more concurrency

### Data Scaling

**Current**: ~10-50K document chunks

**Qdrant Capacity**:
- Millions of vectors
- Add filters for faster search
- Implement pagination for large result sets

## Security Considerations

**Current State**: Development/Local use

**Production Hardening**:
1. Add authentication (JWT tokens)
2. Rate limiting (API endpoints)
3. Input sanitization
4. HTTPS/TLS
5. Network isolation (Docker networks)
6. Secret management (vault)

## Monitoring & Observability

**Current**:
- Health check endpoint
- Stats endpoint
- Docker logs

**Production Additions**:
1. Prometheus metrics
2. Grafana dashboards
3. Structured logging (JSON)
4. Distributed tracing
5. Error tracking (Sentry)

## Future Enhancements

1. **Streaming Responses**: Real-time token generation
2. **Multi-modal**: Support images, diagrams
3. **Fine-tuning**: Customize models on your data
4. **Agent Mode**: Tool use, web search integration
5. **Caching**: Cache common queries
6. **A/B Testing**: Compare different models/prompts
