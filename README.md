# DevOps AI Assistant - Local LLM with RAG

A production-ready AI assistant powered by local LLMs (Ollama) with Retrieval-Augmented Generation (RAG) for DevOps and programming documentation. Query Kubernetes, Terraform, Docker, Python, Go, Bash, and other tools using natural language.

## Features

- **Local LLM Inference**: Ollama with support for multiple models (Llama 3.1, Mistral, Qwen2.5-Coder, etc.)
- **Advanced RAG Pipeline**: Multi-stage retrieval with hybrid search, reranking, and query expansion
- **Web Search Fallback**: Automatic Tavily web search when local docs don't have the answer
- **30+ Documentation Sources**: Complete DevOps stack (K8s, Docker, Terraform, ELK, Grafana) + 6 programming languages + CI/CD tools
- **Web UI**: Clean, responsive chat interface with Dark and Catppuccin Mocha themes
- **AI Coding Assistant**: Aider integration with Qwen2.5-Coder for AI pair programming
- **REST API**: FastAPI backend for integration with other tools
- **Observability**: Prometheus metrics, Grafana dashboards, retrieval analytics
- **Extensible**: MCP and n8n integration for workflow automation
- **GPU Acceleration**: Optimized for NVIDIA GPUs (tested on RTX 3090 24GB)
- **Document Ingestion**: Automated pipeline to scrape and index documentation
- **Conversation Memory**: Redis-backed chat history
- **Docker Hub Ready**: Pre-built images available for instant deployment

## System Requirements

- **CPU**: AMD Ryzen 9 9950X or similar (16+ cores recommended)
- **GPU**: NVIDIA RTX 3090 24GB (or any GPU with 16GB+ VRAM)
- **RAM**: 128GB (32GB minimum)
- **Storage**: 100GB+ SSD for models and vector DB
- **OS**: Linux (Ubuntu 22.04, 24.04, 25.04 tested), Docker & Docker Compose

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Web UI    │─────▶│  FastAPI     │─────▶│   Ollama    │
│  (React)    │      │   Backend    │      │   (LLM)     │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ├──────────────▶┌─────────────┐
                            │               │   Qdrant    │
                            │               │  (Vectors)  │
                            │               └─────────────┘
                            │
                            └──────────────▶┌─────────────┐
                                            │    Redis    │
                                            │  (Memory)   │
                                            └─────────────┘
```

## Quick Start

### Option 1: Using Pre-built Docker Hub Images (Fastest)

```bash
# Clone the repository
git clone https://github.com/jconover/ai-rag-stack.git
cd ai-rag-stack

# Pull and start all services (uses pre-built images from Docker Hub)
docker compose pull
docker compose up -d

# Wait for Ollama to be healthy, then pull your preferred model
docker exec ollama ollama pull llama3.1:8b

# Ingest DevOps documentation
python scripts/ingest_docs.py

# Access the UI
open http://localhost:3000
```

### Option 2: Build from Source (For Development)

```bash
# Clone the repository
git clone https://github.com/jconover/ai-rag-stack.git
cd ai-rag-stack

# Use the dev compose file with local builds
docker compose -f docker-compose.dev.yml up -d --build

# Pull your preferred model
docker exec ollama ollama pull llama3.1:8b

# Ingest DevOps documentation
python scripts/ingest_docs.py

# Access the UI
open http://localhost:3000
```

## Available Models

### For Chat & Documentation

Recommended models for your hardware:

- **llama3.1:8b** - Best general purpose (8GB VRAM)
- **mistral:7b** - Fast and efficient (7GB VRAM)
- **deepseek-coder:6.7b** - Smaller coding model

```bash
# Pull additional models
docker exec ollama ollama pull mistral:7b
```

### For Coding Assistant (Aider)

- **qwen2.5-coder:7b** - Fast coding assistant (~4.7GB VRAM) ⚡
- **qwen2.5-coder:32b** - Powerful coding assistant (~19GB VRAM) 🚀

```bash
# Setup coding assistant (installs Aider + pulls models)
make setup-aider
```

## Documentation Sources

The ingestion pipeline automatically indexes **30+ comprehensive documentation sources**:

### DevOps & Infrastructure
- **Kubernetes**: Official K8s docs (concepts, reference, tutorials)
- **Terraform**: HashiCorp Terraform configuration and providers
- **Docker**: Docker Engine, Compose, Swarm documentation
- **Ansible**: Playbooks, roles, modules, and best practices
- **Helm**: Kubernetes package manager documentation

### Monitoring & Observability
- **Prometheus**: Metrics collection, queries, and alerting
- **Grafana**: Dashboards, data sources, and visualization
- **ELK Stack**: Elasticsearch, Logstash, Kibana (full stack)

### Programming Languages
- **Python**: Official stdlib and language reference
- **Go**: Language spec, packages, and effective Go
- **Rust**: Official docs + Rust by Example
- **JavaScript/Node.js**: Node.js API + MDN JavaScript reference
- **Bash**: GNU Bash manual and scripting guides
- **Zsh**: Shell manual, completions, and hooks

### CI/CD & GitOps
- **Git**: Pro Git book + official documentation
- **Jenkins**: Pipeline configuration and plugins
- **GitHub Actions**: Workflows and automation
- **ArgoCD**: GitOps continuous delivery
- **GitLab CI/CD**: Pipeline configuration

### Cloud Platforms
- **AWS**: EC2, S3, Lambda, ECS, VPC, IAM
- **Azure**: AKS, DevOps, Container Instances
- **GCP**: Google Cloud Platform services and APIs

### Automation & Integration
- **n8n**: Workflow automation and integration
- **JSON Schema**: Configuration validation
- **YAML**: Specification and best practices

### Custom Documentation
- **Custom Docs**: Add your own markdown/text files to `data/custom/`

## Advanced RAG Features

The RAG pipeline includes several advanced features for improved retrieval quality:

### Hybrid Search (BM25 + Vector)

Combines semantic vector search with keyword matching for better results on technical queries:

```bash
# Enable in docker-compose or .env
HYBRID_SEARCH_ENABLED=true
HYBRID_SEARCH_ALPHA=0.5  # Balance between vector (1.0) and keyword (0.0)
```

### HyDE Query Expansion

Generates hypothetical documents for vague queries to improve semantic matching:

```bash
# Enable HyDE
HYDE_ENABLED=true
HYDE_MODEL=llama3.1:8b
```

Smart skip patterns avoid HyDE for specific queries (CLI commands, error messages, file paths).

### Cross-Encoder Reranking

Reranks initial results using a cross-encoder model for improved relevance:

```bash
RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Web Search Fallback (Tavily)

When local documentation doesn't have the answer (low similarity scores), the system automatically falls back to web search:

```bash
# Enable web search fallback
WEB_SEARCH_ENABLED=true
TAVILY_API_KEY=tvly-your-api-key  # Get free key at https://tavily.com

# Configuration
WEB_SEARCH_MIN_SCORE_THRESHOLD=0.4  # Trigger when avg score below this
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_INCLUDE_DOMAINS=docs.aws.amazon.com,kubernetes.io,docs.docker.com
```

**How it works:**
1. Query runs through local vector search + reranking
2. If average similarity score < threshold (default 0.4), web search triggers
3. Tavily searches trusted documentation domains
4. Results are merged into LLM context with source attribution

**Cost:** Free tier includes 1,000 searches/month. Pay-as-you-go: $0.008/search.

**Example response metrics:**
```json
{
  "retrieval_metrics": {
    "hybrid_search_used": true,
    "hyde_used": true,
    "reranker_used": true,
    "web_search_used": true,
    "web_search_reason": "low_avg_score_0.007",
    "web_search_results": 5,
    "web_search_time_ms": 450.32
  }
}
```

## Keeping Documentation Updated

Your documentation stays fresh with automated updates:

### Manual Updates

Update all documentation to latest versions:

```bash
# Update existing documentation repositories
make update-docs

# This will:
# 1. Pull latest changes from all git repositories
# 2. Show which repos were updated
# 3. Automatically re-ingest if updates found
```

### Automated Updates (n8n)

Set up automated weekly or nightly updates:

**Weekly Updates (Recommended):**
- Runs every Sunday at 2 AM
- Full notifications via Slack/Email
- Best for most use cases

**Nightly Updates:**
- Runs every day at 2 AM
- Silent mode (only notifies on updates)
- Best for bleeding-edge environments

**Setup:**
```bash
# 1. Add n8n to docker-compose.yml (see n8n-workflows/README.md)
docker compose up -d n8n

# 2. Import workflow at http://localhost:5678
#    - weekly-doc-update.json (recommended)
#    - daily-doc-update.json (high frequency)

# 3. Configure Slack/Email notifications
# 4. Activate workflow
```

See [n8n-workflows/README.md](n8n-workflows/README.md) for complete setup instructions.

## Project Structure

```
ai-rag-stack/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── main.py         # API entry point
│   │   ├── rag.py          # RAG pipeline
│   │   ├── vectorstore.py  # Qdrant client
│   │   └── models.py       # Pydantic models
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/               # React web UI
│   ├── src/
│   │   ├── App.tsx
│   │   └── components/
│   ├── package.json
│   └── Dockerfile
├── scripts/
│   ├── ingest_docs.py     # Documentation ingestion
│   ├── download_docs.sh   # Download documentation
│   └── update_docs.sh     # Update existing docs
├── data/
│   ├── docs/              # Downloaded documentation
│   └── custom/            # Your custom docs
├── n8n-workflows/         # Automation workflows
│   ├── README.md          # Setup instructions
│   ├── weekly-doc-update.json
│   └── daily-doc-update.json
├── docker-compose.yml
├── .env.example
└── README.md
```

## API Endpoints

- `POST /api/chat` - Send a message and get AI response
- `GET /api/models` - List available Ollama models
- `POST /api/ingest` - Ingest new documents
- `GET /api/health` - Health check
- `GET /api/stats` - Vector database statistics

## Usage Examples

```bash
# Query via API
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I create a Kubernetes deployment with 3 replicas?",
    "model": "llama3.1:8b"
  }'

# Check available models
curl http://localhost:8000/api/models

# Get vector DB stats
curl http://localhost:8000/api/stats
```

## Performance Tuning

### GPU Configuration

The stack automatically uses your NVIDIA GPU. Monitor with:

```bash
watch -n 1 nvidia-smi
```

### Ollama Configuration

Edit `docker-compose.yml` to adjust:

```yaml
environment:
  - OLLAMA_NUM_GPU=1
  - OLLAMA_NUM_THREAD=16  # Match your CPU cores
  - OLLAMA_MAX_LOADED_MODELS=2
```

### Vector Database

Qdrant configuration in `backend/app/vectorstore.py`:

- **Chunk size**: 1000 tokens (adjustable for context)
- **Overlap**: 200 tokens
- **Top K results**: 5 (increase for more context)

## Development

### Local Development Setup

```bash
# Use the dev compose file for local development with hot-reload
docker compose -f docker-compose.dev.yml up -d --build

# Backend development (without Docker)
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend development (without Docker)
cd frontend
npm install
npm run dev

# Run Ollama locally (without Docker)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

## Publishing to Docker Hub

If you're forking this project and want to publish your own images:

```bash
# Login to Docker Hub
docker login

# Build, tag, and push images (replace version as needed)
./scripts/push_to_dockerhub.sh v1.0.0

# Or just push latest
./scripts/push_to_dockerhub.sh
```

The script will:
1. Build both backend and frontend images
2. Tag them with your version and 'latest'
3. Push to Docker Hub
4. Display the image URLs

**Docker Hub Images:**
- Backend: [jconover/ai-rag-backend](https://hub.docker.com/r/jconover/ai-rag-backend)
- Frontend: [jconover/ai-rag-frontend](https://hub.docker.com/r/jconover/ai-rag-frontend)

## AI Coding Assistant (Aider)

This project includes **Aider**, an AI pair programming tool that works with your local Ollama models.

### Features

- **Code Generation**: Write new features with AI assistance
- **Refactoring**: Improve existing code with smart suggestions
- **Bug Fixes**: Get help debugging and fixing issues
- **Git Integration**: Auto-commits changes with descriptive messages
- **Multi-file Editing**: Works across your entire codebase
- **Local Models**: Uses Qwen2.5-Coder via Ollama (no API keys needed)

### Setup

```bash
# One-time setup: Install Aider and pull models
make setup-aider
```

This will:
1. Install Aider CLI tool
2. Pull `qwen2.5-coder:7b` (fast, 4.7GB)
3. Pull `qwen2.5-coder:32b` (powerful, 19GB)

### Usage

```bash
# Start Aider with 7B model (faster responses)
make aider

# Start Aider with 32B model (more capable)
make aider-32b

# Or run directly with custom options
aider --model ollama/qwen2.5-coder:7b --edit-format diff
```

### Example Session

```bash
$ make aider
Aider v0.59.0
Model: qwen2.5-coder:7b with diff edit format
Git repo: /home/user/ai-rag-stack

> Add a new API endpoint to export chat history as JSON

# Aider will:
# 1. Analyze your codebase
# 2. Edit the necessary files
# 3. Auto-commit the changes
# 4. Show you a diff of what changed
```

### Tips

- **7B model**: Use for quick edits, bug fixes, and simple features
- **32B model**: Use for complex refactoring, architecture changes, and new features
- **Both run simultaneously**: Your RTX 3090 can handle both models in VRAM!
- **Git integration**: Aider auto-commits, so you can easily review and rollback changes

### Configuration

Two config files are provided:
- `.aider.conf.yml` - 7B model configuration
- `.aider.32b.conf.yml` - 32B model configuration

Customize them to adjust:
- Max tokens
- Edit format (diff, whole, or udiff)
- Auto-commit behavior
- Dark mode settings

## MCP and n8n Integration

This RAG stack can be extended with powerful automation and integration capabilities:

### MCP (Model Context Protocol)
Expose your RAG system to Claude Desktop and other AI tools:
- **Tool Calling**: Let Claude search your documentation
- **External Sources**: Integrate Jira, Confluence, Slack
- **Custom Actions**: Execute kubectl, terraform commands safely

### n8n Workflow Automation
Automate DevOps workflows:
- **Slack Bot**: Auto-answer questions in Slack channels
- **Incident Response**: Auto-query runbooks during alerts
- **CI/CD Helper**: Comment on failed pipelines with solutions
- **Documentation Updates**: Auto-sync docs weekly or nightly with notifications

**Pre-built workflows available in `n8n-workflows/`:**
- `weekly-doc-update.json` - Weekly updates with full notifications (recommended)
- `daily-doc-update.json` - Nightly updates with silent mode

**See:**
- [MCP_N8N_INTEGRATION.md](MCP_N8N_INTEGRATION.md) - Integration guides and examples
- [n8n-workflows/README.md](n8n-workflows/README.md) - Setup and usage instructions

## Troubleshooting

### Ollama won't start
- Check GPU drivers: `nvidia-smi`
- Ensure Docker has GPU access: `docker run --gpus all nvidia/cuda:12.0-base nvidia-smi`

### Out of memory errors
- Use smaller models (7B-13B instead of 70B)
- Reduce context window in `backend/app/rag.py`
- Adjust `OLLAMA_MAX_LOADED_MODELS` to 1

### Slow responses
- Switch to faster model (mistral:7b)
- Reduce Top K results in vector search
- Use SSD for vector database storage

## Contributing

This is a portfolio project, but contributions are welcome!

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM inference
- [Qdrant](https://qdrant.tech/) - Vector database
- [LangChain](https://langchain.com/) - RAG framework
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
