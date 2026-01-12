# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DevOps AI Assistant** - A RAG (Retrieval-Augmented Generation) system using local LLMs via Ollama to answer questions about DevOps documentation. 30+ documentation sources indexed in Qdrant vector database.

**Tech Stack**: FastAPI (Python) + React + Ollama (LLM) + Qdrant (vector DB) + Redis (conversation memory)

## Common Commands

```bash
# Start/Stop
make start              # Production (Docker Hub images)
make start-dev          # Development (local builds + hot-reload)
make stop               # Stop all services
make logs-backend       # View backend logs

# First-time setup
make pull-model         # Pull llama3.1:8b (default)
make pull-model MODEL=mistral:7b  # Pull specific model
make ingest             # Download and index all documentation

# Operations
make health             # Service health check
make stats              # Vector DB statistics
make update-docs        # Update docs to latest versions

# AI Coding (Aider)
make setup-aider        # Install Aider + Qwen2.5-Coder
make aider              # Start with qwen2.5-coder:7b
make aider-32b          # Start with qwen2.5-coder:32b
```

### Testing API Endpoints
```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I create a K8s deployment?", "model": "llama3.1:8b"}'

# Streaming response
curl --no-buffer -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain Kubernetes pods", "model": "llama3.1:8b"}'
```

## Architecture

```
Frontend (React:3000) → Backend (FastAPI:8000) → Ollama (LLM:11434)
                              ↓                      Qdrant (Vector:6333)
                              ↓                      Redis (Memory:6379)
```

### RAG Pipeline Flow
1. User query → Backend `/api/chat` or `/api/chat/stream`
2. **Conversation context** (optional): Resolve pronouns ("it", "that") using prior messages
3. **HyDE expansion** (optional): Generate hypothetical document for vague queries
4. **Hybrid search**: BM25 keyword + vector semantic search with RRF fusion
5. **Reranking**: Cross-encoder reranks top 20 → top 5
6. **Web search fallback**: If scores low, Tavily searches trusted doc sites
7. Context + query → Ollama LLM → Response with source attribution
8. Conversation saved to Redis (24h TTL)

### Key Files

| File | Purpose |
|------|---------|
| `backend/app/rag.py` | RAG pipeline: `generate_response()`, `generate_response_stream()` |
| `backend/app/main.py` | FastAPI endpoints: `/api/chat`, `/api/chat/stream`, `/api/upload` |
| `backend/app/vectorstore.py` | Qdrant interface, hybrid search (`VectorStore` class) |
| `backend/app/reranker.py` | Cross-encoder reranking |
| `backend/app/query_expansion.py` | HyDE query expansion |
| `backend/app/conversation_context.py` | Conversation-aware query expansion for follow-ups |
| `backend/app/web_search.py` | Tavily web search fallback |
| `backend/app/config.py` | Environment configuration (reads `.env`) |
| `scripts/ingest_docs.py` | Document ingestion pipeline |
| `scripts/download_docs.sh` | Git clone documentation repos |

### Document Ingestion Flow
```
Markdown/Text → LangChain DirectoryLoader → RecursiveCharacterTextSplitter
(chunk_size=1000, overlap=200) → HuggingFace Embeddings → Qdrant
```

## Implementation Details

### RAG Configuration
- **Chunk Size**: 1000 chars with 200 overlap
- **Embedding**: `BAAI/bge-base-en-v1.5` (768 dims, CPU)
- **Top K**: 5 documents per query
- **Collection**: `devops_docs` in Qdrant

> **Note**: Changing the embedding model requires full re-ingestion of all documents. Run `make ingest` after modifying `EMBEDDING_MODEL` or `EMBEDDING_DIMENSION` in `.env`.

### Environment Variables (`.env`)

**Core Services:**
- `OLLAMA_HOST` - Ollama endpoint (default: http://ollama:11434)
- `QDRANT_HOST`, `QDRANT_PORT` - Vector DB
- `REDIS_HOST`, `REDIS_PORT` - Conversation memory
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_RESULTS` - RAG tuning
- `EMBEDDING_MODEL` - HuggingFace embedding model (default: BAAI/bge-base-en-v1.5)
- `EMBEDDING_DIMENSION` - Vector dimension matching model output (default: 768)

**Advanced RAG Features:**
- `HYBRID_SEARCH_ENABLED=true` - Enable BM25 + vector hybrid search
- `RERANKER_ENABLED=true` - Enable cross-encoder reranking
- `HYDE_ENABLED=true` - Enable HyDE query expansion
- `CONVERSATION_CONTEXT_ENABLED=true` - Enable conversation-aware retrieval (default: true)
- `CONVERSATION_CONTEXT_HISTORY_LIMIT=3` - Number of prior messages to consider
- `CONVERSATION_CONTEXT_MAX_TERMS=10` - Max context terms to extract from history
- `WEB_SEARCH_ENABLED=true` - Enable Tavily web search fallback
- `TAVILY_API_KEY=tvly-xxx` - Tavily API key (free: 1,000/month)
- `WEB_SEARCH_MIN_SCORE_THRESHOLD=0.4` - Trigger web search below this score
- `WEB_SEARCH_INCLUDE_DOMAINS=docs.aws.amazon.com,kubernetes.io` - Trusted domains

### Docker
- **Production**: `docker-compose.yml` (Docker Hub images)
- **Development**: `docker-compose.dev.yml` (local builds, hot-reload)
- **GPU**: Ollama uses NVIDIA GPU via `runtime: nvidia`

## Modifying the System

### Adding Documentation Sources
1. Edit `scripts/download_docs.sh` to clone new repo
2. Add source to `doc_sources` dict in `scripts/ingest_docs.py`
3. Run `make download-docs && make ingest`

### Adding API Endpoints
```python
# 1. Define Pydantic models in models.py
class NewFeatureRequest(BaseModel):
    param: str

# 2. Add endpoint in main.py
@app.post("/api/new-feature")
async def new_feature(request: NewFeatureRequest):
    result = some_logic(request.param)
    return {"result": result}
```

### Adding Prompt Templates
```python
# Edit backend/app/templates.py - add to PROMPT_TEMPLATES list
{
    "id": "my-template",
    "category": "MyCategory",
    "title": "My Template Title",
    "description": "What this template does",
    "prompt": "The actual prompt text"
}
```

### Changing LLM Prompt
Modify `backend/app/rag.py` → `_build_prompt()` method

## Debugging

```bash
make stats              # Check vector DB has data (points_count > 0)
make health             # Verify all services connected
docker exec ollama ollama list  # Check available models
docker compose restart backend  # Restart after code changes
```

### Performance Tuning
- **Faster**: smaller models (mistral:7b) or lower `TOP_K_RESULTS`
- **Better quality**: larger models or higher `TOP_K_RESULTS`
- **Memory issues**: reduce `OLLAMA_MAX_LOADED_MODELS` in docker-compose.yml

## Code Conventions

- Python: async/await for I/O, type hints, HTTPException for errors
- Singletons for shared resources: `rag_pipeline`, `vector_store`
- Configuration centralized in `config.py` via environment variables

## Claude Code Agents

This repo includes 12 specialized AI/ML subagents in `.claude/agents/` for comprehensive code reviews:

| Agent | Focus |
|-------|-------|
| `ai-engineer` | AI system design, production deployment |
| `llm-architect` | LLM architecture, RAG optimization |
| `ml-engineer` | ML lifecycle, model serving |
| `mlops-engineer` | ML infrastructure, CI/CD |
| `prompt-engineer` | Prompt design, optimization |
| `data-engineer` | Data pipelines, ETL |
| `data-scientist` | Statistical analysis, modeling |
| `database-optimizer` | Query optimization, indexing |
| `nlp-engineer` | NLP pipelines, embeddings |
| `data-analyst` | Analytics, visualization |
| `postgres-pro` | Database administration |
| `machine-learning-engineer` | Inference, edge deployment |

**Usage:** Run parallel expert reviews with:
```
Launch 12 agents to review this project from different perspectives
```

*Credit: Agents from [awesome-claude-code-subagents](https://github.com/VoltAgent/awesome-claude-code-subagents) (categories/05-data-ai)*

## Known Limitations

- Embedding model runs on CPU (change to CUDA in vectorstore.py for GPU)
- No authentication (add JWT if deploying publicly)
- Redis conversation history expires after 24h
- Vector search limited to top 5 results (configurable via `TOP_K_RESULTS`)
