# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **DevOps AI Assistant** - a production-ready RAG (Retrieval-Augmented Generation) system that uses local LLMs via Ollama to answer questions about DevOps and programming documentation. The system includes 30+ documentation sources (Kubernetes, Terraform, Docker, Python, Go, Rust, etc.) indexed in a vector database.

**Tech Stack**: FastAPI (Python backend) + React (frontend) + Ollama (LLM) + Qdrant (vector DB) + Redis (conversation memory)

## Recent Features (Added 2025-11)

### 1. Streaming Responses
- **Real-time token streaming** using Server-Sent Events (SSE)
- Toggle between streaming and batch mode in UI
- Backend: `/api/chat/stream` endpoint with `generate_response_stream()` in `rag.py`
- Frontend: Fetch API with ReadableStream for progressive rendering

### 2. Prompt Templates
- **15 pre-built templates** for common DevOps tasks (K8s, Terraform, Docker, etc.)
- Categories: Kubernetes, Terraform, Docker, Ansible, Monitoring, CI/CD, Debugging, Security, Scripting, Operations
- Backend: `templates.py` module with `/api/templates` endpoint
- Frontend: Modal UI with searchable template cards

### 3. Document Upload via UI
- **Upload custom documentation** (.md, .txt, .markdown) through web interface
- Automatic ingestion into vector database
- Backend: `/api/upload` endpoint with multipart/form-data support
- Frontend: Drag-and-drop file upload with progress indicators

## Common Commands

### Development
```bash
# Production (uses Docker Hub images)
make start              # Start all services
make stop               # Stop all services
make logs               # View all logs
make logs-backend       # View backend logs only

# Development (builds locally)
make start-dev          # Start with local builds and hot-reload

# After first start, pull a model and ingest docs
make pull-model         # Pulls llama3.1:8b by default
make pull-model MODEL=mistral:7b  # Pull specific model
make ingest             # Download and index all documentation
```

### Useful Operations
```bash
make health             # Check service health
make stats              # Vector DB statistics
make list-models        # List available Ollama models
make update-docs        # Update documentation to latest versions
```

### AI Coding Assistant (Aider)
```bash
make setup-aider        # One-time setup (installs Aider + Qwen2.5-Coder)
make setup-aider-deepseek # Setup with DeepSeek Coder models (1.3b, 6.7b, 33b)
make aider              # Start with qwen2.5-coder:7b (fast)
make aider-32b          # Start with qwen2.5-coder:32b (powerful)
make aider-deepseek     # Start with deepseek-coder:6.7b
make aider-deepseek-33b # Start with deepseek-coder:33b (most powerful)
```

### Testing
```bash
# Test API manually
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I create a K8s deployment?", "model": "llama3.1:8b"}'

# Test new features
curl http://localhost:8000/api/templates  # Get prompt templates
curl -X POST http://localhost:8000/api/upload \  # Upload custom doc
  -F "files=@path/to/doc.md" \
  -F "auto_ingest=true"

# Test streaming (using curl with --no-buffer to see real-time output)
curl --no-buffer -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain Kubernetes pods", "model": "llama3.1:8b"}'
```

## Architecture

### Service Architecture
```
Frontend (React:3000) → Backend (FastAPI:8000) → Ollama (LLM:11434)
                              ↓                      Qdrant (Vector:6333)
                              ↓                      Redis (Memory:6379)
```

### RAG Pipeline Flow
1. **User Query** → Backend receives via `/api/chat`
2. **Vector Search** → Query embedded using `sentence-transformers/all-MiniLM-L6-v2` (384 dims), searches Qdrant for top 5 similar chunks
3. **Context Building** → Retrieved docs formatted into prompt context
4. **LLM Generation** → Ollama generates response using context
5. **Response** → Returned to user with source attribution, saved to Redis

### Key Components

**Backend (`backend/app/`):**
- `main.py` - FastAPI app, API endpoints:
  - `/api/chat` - Standard chat (batch response)
  - `/api/chat/stream` - Streaming chat (SSE)
  - `/api/templates` - Get prompt templates
  - `/api/upload` - Upload custom docs
  - `/api/health`, `/api/models`, `/api/stats` - System info
- `rag.py` - RAG pipeline orchestration (`RAGPipeline` class)
  - `generate_response()` - Batch generation
  - `generate_response_stream()` - Streaming generation
- `vectorstore.py` - Qdrant interface (`VectorStore` class)
- `templates.py` - Prompt template definitions (15 templates)
- `config.py` - Environment configuration
- `models.py` - Pydantic request/response models

**Frontend (`frontend/src/`):**
- `App.js` - Main chat interface component
- React + Axios for API calls, deployed via Nginx on port 3000

**Scripts (`scripts/`):**
- `ingest_docs.py` - Documentation ingestion pipeline (markdown → chunks → embeddings → Qdrant)
- `download_docs.sh` - Git clones documentation repositories
- `update_docs.sh` - Updates existing docs and triggers re-ingestion if changed

### Data Flow for Document Ingestion
```
Markdown/Text Files → LangChain DirectoryLoader → RecursiveCharacterTextSplitter
(chunk_size=1000, overlap=200) → HuggingFace Embeddings → Qdrant Vector Store
```

## Important Implementation Details

### RAG Configuration
- **Chunk Size**: 1000 characters (~250 tokens) with 200 char overlap
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, CPU)
- **Top K**: 5 documents retrieved per query
- **Collection**: `devops_docs` in Qdrant
- **Distance Metric**: Cosine similarity

### Environment Variables
Critical env vars (defined in `.env`, read by `backend/app/config.py`):
- `OLLAMA_HOST` - Ollama API endpoint (default: http://ollama:11434)
- `QDRANT_HOST`, `QDRANT_PORT` - Vector DB connection
- `REDIS_HOST`, `REDIS_PORT` - Conversation memory
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_RESULTS` - RAG tuning parameters

### Conversation Memory
- Stored in Redis with key pattern `chat:{session_id}`
- TTL: 24 hours
- Stores list of `{role, content}` message objects

### Docker Architecture
- **Production**: `docker-compose.yml` uses pre-built images from Docker Hub
- **Development**: `docker-compose.dev.yml` builds locally with hot-reload
- **GPU**: Ollama container uses NVIDIA GPU via `deploy.resources.reservations.devices`
- **Healthchecks**: All services have healthcheck definitions

## Modifying the System

### Adding New Documentation Sources
1. Edit `scripts/download_docs.sh` to clone new repo
2. Add source to `doc_sources` dict in `scripts/ingest_docs.py`
3. Run `make download-docs && make ingest`

### Adjusting RAG Parameters
Edit `.env` (or environment in `docker-compose.yml`):
- `CHUNK_SIZE` - Larger = more context per chunk (default 1000)
- `CHUNK_OVERLAP` - Prevents context loss at boundaries (default 200)
- `TOP_K_RESULTS` - More = more context to LLM (default 5)

### Adding API Endpoints
Add to `backend/app/main.py`, use existing patterns:
- Async endpoint definitions
- Pydantic models for validation (defined in `models.py`)
- Error handling with HTTPException

### Changing LLM Behavior
Modify prompt template in `backend/app/rag.py` → `_build_prompt()` method

## Testing Strategy

When making changes:
1. **Backend Changes**: Restart with `docker compose restart backend`, check logs with `make logs-backend`
2. **Frontend Changes**: In dev mode, changes auto-reload. In prod, rebuild: `docker compose up -d --build frontend`
3. **RAG Changes**: Test via API first, then UI
4. **Verify Health**: `make health` should show all services connected

## Common Patterns

### Using the New Features

**Streaming Responses:**
```python
# In backend/app/rag.py - already implemented
async def generate_response_stream(self, query, model, ...):
    # Yields chunks: {'type': 'metadata'|'content'|'done'|'error', ...}
    yield {'type': 'metadata', 'sources': [...]}
    for chunk in ollama.chat(..., stream=True):
        yield {'type': 'content', 'content': chunk['message']['content']}
    yield {'type': 'done'}
```

**Adding New Prompt Templates:**
```python
# Edit backend/app/templates.py - add to PROMPT_TEMPLATES list
{
    "id": "my-template",
    "category": "MyCategory",
    "title": "My Template Title",
    "description": "What this template does",
    "prompt": "The actual prompt text with placeholders"
}
```

**Uploading Documents Programmatically:**
```bash
# Upload multiple files at once
curl -X POST http://localhost:8000/api/upload \
  -F "files=@doc1.md" \
  -F "files=@doc2.md" \
  -F "auto_ingest=true"
```

### Adding a New Backend Feature
```python
# 1. Define Pydantic models in models.py
class NewFeatureRequest(BaseModel):
    param: str

# 2. Add endpoint in main.py
@app.post("/api/new-feature")
async def new_feature(request: NewFeatureRequest):
    result = some_logic(request.param)
    return {"result": result}

# 3. Test via curl or frontend
```

### Debugging RAG Issues
1. Check vector DB has data: `make stats` (should show points_count > 0)
2. Test search: Query `/api/stats` to verify collection exists
3. Check Ollama: `docker exec ollama ollama list` to see models
4. Verify connections: `/api/health` endpoint

### Performance Tuning
- **Faster responses**: Use smaller models (mistral:7b) or reduce TOP_K_RESULTS
- **Better quality**: Use larger models (llama3.1:70b) or increase TOP_K_RESULTS
- **Memory issues**: Reduce `OLLAMA_MAX_LOADED_MODELS` in docker-compose.yml
- **Slow ingestion**: Increase `MAX_WORKERS` in ingest_docs.py

## Important Conventions

### Code Style
- Python: Follow PEP 8, use type hints, async/await for I/O operations
- Use singleton patterns for shared resources (`rag_pipeline`, `vector_store`)
- Configuration via environment variables, centralized in `config.py`

### Error Handling
- Backend: Use HTTPException for API errors
- Log errors but don't expose internals to users
- Health checks should gracefully handle service failures

### Docker Best Practices
- Use healthchecks for service dependencies
- Volume mount `./data` and `./scripts` for easy access
- Use named volumes for persistence (ollama_data, qdrant_data, redis_data)

## Documentation Updates

### Automated Updates (n8n)
- Pre-built workflows in `n8n-workflows/` directory
- `weekly-doc-update.json` - Recommended for most use cases
- `daily-doc-update.json` - For bleeding-edge environments
- See `MCP_N8N_INTEGRATION.md` for setup details

### Manual Updates
```bash
make update-docs  # Pulls latest from git repos, re-ingests if changed
```

## Key Files to Understand

When modifying the system, these are the most important files:
1. `backend/app/rag.py` - Core RAG logic
2. `backend/app/main.py` - API endpoints
3. `scripts/ingest_docs.py` - Document processing
4. `docker-compose.yml` - Service orchestration
5. `Makefile` - Common operations

## Known Limitations

- Embedding model runs on CPU (change to CUDA in vectorstore.py/ingest_docs.py for GPU)
- No authentication/authorization (add JWT if deploying publicly)
- No streaming responses (Ollama supports it, not implemented)
- Redis conversation history expires after 24h
- Vector search is limited to top 5 results (configurable)

## Resources

- **API Docs**: http://localhost:8000/docs (FastAPI auto-generated)
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Architecture Details**: See `ARCHITECTURE.md`
- **Setup Guide**: See `README.md` and `SETUP.md`
- **MCP/n8n Integration**: See `MCP_N8N_INTEGRATION.md`
