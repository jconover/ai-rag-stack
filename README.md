# DevOps AI Assistant - Local LLM with RAG

A production-ready AI assistant powered by local LLMs (Ollama) with Retrieval-Augmented Generation (RAG) for DevOps documentation. Query Kubernetes, Terraform, Docker, Ansible, and other DevOps tools using natural language.

## Features

- **Local LLM Inference**: Ollama with support for multiple models (Llama 3.1, Mistral, CodeLlama, etc.)
- **RAG Pipeline**: Vector search using Qdrant for accurate, context-aware responses
- **DevOps Documentation**: Pre-configured to ingest K8s, Terraform, Docker, Ansible, AWS, and more
- **Web UI**: Clean, responsive chat interface
- **REST API**: FastAPI backend for integration with other tools
- **GPU Acceleration**: Optimized for NVIDIA GPUs (tested on RTX 3090 24GB)
- **Document Ingestion**: Automated pipeline to scrape and index documentation
- **Conversation Memory**: Redis-backed chat history

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

```bash
# Navigate to project directory
git clone https://github.com/jconover/ai-rag-stack.git
cd ai-rag-stack

# Verify your system is ready (checks Docker, GPU, disk space, etc.)
bash scripts/verify_setup.sh

# Initial setup
make setup

# Start all services (Ollama, Qdrant, Redis, API, UI)
make start

# Pull your preferred model (run this after Ollama starts)
docker exec ollama ollama pull llama3.1:8b

# Ingest DevOps documentation
python scripts/ingest_docs.py

# Access the UI
open http://localhost:3000
```

## Available Models

Recommended models for your hardware:

- **llama3.1:8b** - Best general purpose (8GB VRAM)
- **codellama:13b** - Better for code generation (13GB VRAM)
- **llama3.1:70b** - Most capable, slower (40GB+ VRAM, requires quantization)
- **mistral:7b** - Fast and efficient (7GB VRAM)
- **deepseek-coder:33b** - Excellent for code (20GB+ VRAM)

```bash
# Pull additional models
docker exec ollama ollama pull codellama:13b
docker exec ollama ollama pull mistral:7b
```

## Documentation Sources

The ingestion pipeline automatically indexes:

- **Kubernetes**: Official K8s docs (concepts, reference, tutorials)
- **Terraform**: HashiCorp Terraform docs
- **Docker**: Docker Engine, Compose, Swarm docs
- **Ansible**: Ansible documentation and best practices
- **AWS**: AWS service documentation (EC2, S3, Lambda, ECS, etc.)
- **Azure**: Azure DevOps, AKS, Container Instances
- **GitLab CI/CD**: CI/CD pipeline documentation
- **Prometheus/Grafana**: Monitoring and observability
- **Custom Docs**: Add your own markdown/text files to `data/custom/`

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
│   └── download_docs.sh   # Download documentation
├── data/
│   ├── docs/              # Downloaded documentation
│   └── custom/            # Your custom docs
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

```bash
# Backend development
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend development
cd frontend
npm install
npm run dev

# Run Ollama locally (without Docker)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

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
