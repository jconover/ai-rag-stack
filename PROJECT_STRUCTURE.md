# Project Structure

Complete file tree and description of the DevOps AI Assistant RAG system.

```
ai-rag-stack/
│
├── README.md                          # Main project documentation
├── QUICKSTART_UBUNTU_25.04.md        # Fast setup for Ubuntu 25.04
├── SETUP.md                          # Detailed setup instructions
├── ARCHITECTURE.md                   # System design and data flow
├── CONTRIBUTING.md                   # Contribution guidelines
├── YOUR_SYSTEM_STATUS.md             # Current system status
├── PROJECT_STRUCTURE.md              # This file
├── LICENSE                           # MIT License
│
├── .env.example                      # Environment configuration template
├── .gitignore                        # Git ignore patterns
├── Makefile                          # Convenient command shortcuts
├── docker-compose.yml                # Multi-container orchestration
│
├── backend/                          # Python FastAPI backend
│   ├── Dockerfile                    # Backend container definition
│   ├── requirements.txt              # Python dependencies
│   └── app/
│       ├── __init__.py              # Package initialization
│       ├── main.py                  # FastAPI application & endpoints
│       ├── config.py                # Configuration management
│       ├── models.py                # Pydantic request/response models
│       ├── rag.py                   # RAG pipeline implementation
│       └── vectorstore.py           # Qdrant vector database interface
│
├── frontend/                         # React web UI
│   ├── Dockerfile                   # Frontend container definition
│   ├── nginx.conf                   # Nginx reverse proxy config
│   ├── package.json                 # Node.js dependencies
│   ├── public/
│   │   └── index.html              # HTML entry point
│   └── src/
│       ├── index.js                # React entry point
│       ├── index.css               # Global styles
│       ├── App.js                  # Main chat component
│       └── App.css                 # Component styles
│
├── scripts/                          # Utility scripts
│   ├── verify_setup.sh              # System requirements checker
│   ├── download_docs.sh             # Documentation downloader
│   ├── ingest_docs.py               # Document indexing pipeline
│   └── test_api.sh                  # API endpoint testing
│
└── data/                             # Data storage
    ├── docs/                        # Downloaded documentation
    │   ├── kubernetes/              # K8s docs (git clone)
    │   ├── terraform/               # Terraform docs (git clone)
    │   ├── docker/                  # Docker docs (git clone)
    │   ├── ansible/                 # Ansible docs (git clone)
    │   └── prometheus/              # Prometheus docs (git clone)
    └── custom/                      # User's custom documentation
        └── README.md                # Instructions for adding docs
```

## File Descriptions

### Root Level Documentation

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation with features, setup, and usage |
| `QUICKSTART_UBUNTU_25.04.md` | Optimized quick start guide for your system |
| `SETUP.md` | Comprehensive setup guide with troubleshooting |
| `ARCHITECTURE.md` | Deep dive into system architecture and design |
| `YOUR_SYSTEM_STATUS.md` | Your specific hardware/software status |
| `CONTRIBUTING.md` | Guidelines for contributing to the project |
| `LICENSE` | MIT License |

### Configuration Files

| File | Purpose |
|------|---------|
| `.env.example` | Environment variable template (copy to `.env`) |
| `docker-compose.yml` | Defines all services (Ollama, Qdrant, Redis, API, UI) |
| `Makefile` | Convenient commands (start, stop, logs, etc.) |
| `.gitignore` | Files to exclude from git |

### Backend (Python/FastAPI)

| File | Purpose | Lines |
|------|---------|-------|
| `backend/Dockerfile` | Container image for FastAPI app | ~20 |
| `backend/requirements.txt` | Python dependencies | ~30 |
| `backend/app/__init__.py` | Package marker | ~3 |
| `backend/app/main.py` | API endpoints and application logic | ~150 |
| `backend/app/config.py` | Environment configuration management | ~50 |
| `backend/app/models.py` | Pydantic models for API I/O | ~60 |
| `backend/app/rag.py` | RAG pipeline (retrieve, format, generate) | ~120 |
| `backend/app/vectorstore.py` | Qdrant vector database interface | ~80 |

**Total Backend Code:** ~500 lines

### Frontend (React)

| File | Purpose | Lines |
|------|---------|-------|
| `frontend/Dockerfile` | Multi-stage build for React app | ~15 |
| `frontend/nginx.conf` | Nginx configuration for serving | ~20 |
| `frontend/package.json` | Node.js dependencies | ~30 |
| `frontend/public/index.html` | HTML entry point | ~15 |
| `frontend/src/index.js` | React initialization | ~10 |
| `frontend/src/index.css` | Global styles | ~20 |
| `frontend/src/App.js` | Main chat interface component | ~200 |
| `frontend/src/App.css` | Component styling | ~250 |

**Total Frontend Code:** ~560 lines

### Scripts

| File | Purpose | Lines |
|------|---------|-------|
| `scripts/verify_setup.sh` | Pre-flight system check | ~150 |
| `scripts/download_docs.sh` | Clone documentation repos | ~60 |
| `scripts/ingest_docs.py` | Index docs into Qdrant | ~180 |
| `scripts/test_api.sh` | API endpoint testing | ~80 |

**Total Scripts:** ~470 lines

## Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| Frontend (Nginx) | 3000 | Web UI |
| Backend (FastAPI) | 8000 | REST API |
| Qdrant | 6333 | Vector database HTTP API |
| Qdrant | 6334 | Vector database gRPC API |
| Redis | 6379 | Cache and session storage |
| Ollama | 11434 | LLM inference API |

## Docker Volumes

| Volume | Purpose |
|--------|---------|
| `ollama_data` | Model storage (~5-50GB) |
| `qdrant_data` | Vector database storage (~5-20GB) |
| `redis_data` | Conversation history (~100MB) |

## API Endpoints

### Public Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | API information |
| GET | `/api/health` | Service health check |
| POST | `/api/chat` | Chat with AI assistant |
| GET | `/api/models` | List available models |
| GET | `/api/stats` | Vector DB statistics |
| GET | `/api/history/{id}` | Get conversation history |

## Key Dependencies

### Backend
- **FastAPI** - Web framework
- **Ollama** - LLM client
- **LangChain** - RAG framework
- **Qdrant-client** - Vector database
- **Sentence-transformers** - Embeddings
- **Redis** - Caching

### Frontend
- **React** - UI framework
- **Axios** - HTTP client
- **React-markdown** - Markdown rendering
- **Prism.js** - Code syntax highlighting

## Project Statistics

- **Total Python Code:** ~700 lines
- **Total JavaScript/React Code:** ~560 lines
- **Total Shell Scripts:** ~470 lines
- **Total Documentation:** ~2500 lines
- **Total Configuration:** ~200 lines

**Grand Total:** ~4,430 lines

## Service Dependencies

```
Frontend (React)
    ↓
Backend (FastAPI)
    ↓
    ├─→ Ollama (LLM)
    ├─→ Qdrant (Vector DB)
    └─→ Redis (Cache)
        ↓
    GPU (NVIDIA RTX 3090)
```

## Data Flow

```
User Query (Frontend)
    ↓
Backend API (/api/chat)
    ↓
Vector Search (Qdrant)
    ├─→ Embedding Model (sentence-transformers)
    └─→ Top K Documents
         ↓
    Prompt Building (RAG)
         ↓
    LLM Generation (Ollama)
         ├─→ GPU Inference
         └─→ Response
              ↓
         Save History (Redis)
              ↓
         Return to User
```

## Development Workflow

1. **Setup**: `make setup` - Initialize environment
2. **Start**: `make start` - Launch all services
3. **Develop**: Edit code, services auto-reload
4. **Test**: `make test` - Run API tests
5. **Monitor**: `make logs` - View logs
6. **Health Check**: `make health` - Verify services
7. **Stop**: `make stop` - Shutdown services

## Deployment Workflow

1. **Verify**: `bash scripts/verify_setup.sh`
2. **Pull Models**: `make pull-model`
3. **Download Docs**: `make download-docs`
4. **Ingest**: `make ingest`
5. **Start**: `make start`
6. **Test**: `make test`

## Customization Points

### Add New Documentation Source

1. Edit `scripts/download_docs.sh` to add new git clone
2. Run `make download-docs`
3. Run `make ingest`

### Add Custom Model

1. `docker exec ollama ollama pull <model-name>`
2. Select in UI dropdown

### Adjust RAG Parameters

Edit `.env`:
- `CHUNK_SIZE` - Document chunk size
- `CHUNK_OVERLAP` - Overlap between chunks
- `TOP_K_RESULTS` - Number of docs to retrieve

### Tune Performance

Edit `docker-compose.yml`:
- `OLLAMA_NUM_THREAD` - CPU threads
- `OLLAMA_MAX_LOADED_MODELS` - Concurrent models

## Future Enhancements

Potential additions to the project:
- [ ] Streaming responses
- [ ] Multi-modal support (images)
- [ ] Fine-tuning pipeline
- [ ] Monitoring dashboard (Prometheus/Grafana)
- [ ] Authentication/authorization
- [ ] Multi-user support
- [ ] Conversation search
- [ ] Export conversations
- [ ] Model comparison view
- [ ] Custom RAG strategies

## Maintenance

### Regular Tasks

```bash
# Update models
docker exec ollama ollama pull llama3.1:8b

# Update containers
docker-compose pull
docker-compose up -d

# Clean old data
make clean-all
```

### Backup

```bash
# Backup vector database
docker exec qdrant tar czf - /qdrant/storage > backup.tar.gz

# Backup custom docs
tar czf custom-docs.tar.gz data/custom/
```

## Performance Characteristics

### Resource Usage (Idle)
- CPU: 1-2%
- RAM: 2-3GB
- GPU: 0.5GB VRAM
- Disk: Minimal I/O

### Resource Usage (Active - Llama 8B)
- CPU: 10-20%
- RAM: 4-5GB
- GPU: 5-6GB VRAM, 70-90% utilization
- Response Time: 2-5 seconds

### Throughput
- Sequential queries: ~1 query every 3-5 seconds
- Tokens/second: 25-50 (model dependent)
- Vector search latency: <200ms

## Quality Metrics

### Code Quality
- Type hints in Python
- Pydantic models for validation
- Error handling throughout
- Health checks on all services

### Documentation
- Comprehensive README
- Setup guides
- Architecture documentation
- Inline code comments

### DevOps
- Docker containerization
- Multi-service orchestration
- Environment configuration
- Automated testing

This structure demonstrates modern software engineering practices suitable for a professional portfolio.
