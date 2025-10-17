# AI RAG Stack - Quick Reference

## Essential Commands

### Services
```bash
make start          # Start all services (Docker Hub images)
make start-dev      # Start with local builds
make stop           # Stop all services
make restart        # Restart services
make logs           # View all logs
make health         # Check service health
```

### Documentation
```bash
make download-docs  # Download all documentation
make ingest         # Ingest docs into vector DB
make stats          # Show vector database stats
```

### Models
```bash
make pull-model                    # Pull llama3.1:8b (default)
make pull-model MODEL=mistral:7b   # Pull specific model
make list-models                   # List installed models
```

### AI Coding Assistant
```bash
make setup-aider    # Install Aider + pull Qwen models
make aider          # Start Aider with 7B model (fast)
make aider-32b      # Start Aider with 32B model (powerful)
```

### Docker Hub
```bash
docker login        # Login to Docker Hub
make publish        # Build and push images
```

## URLs

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Qdrant**: http://localhost:6333/dashboard

## Documentation Sources

### DevOps
- Kubernetes, Terraform, Docker, Ansible, Prometheus

### Programming
- Python, Go, Bash, Zsh

### Cloud
- AWS, Azure, GitLab CI/CD

## Example Queries

### DevOps
- "How do I create a Kubernetes deployment?"
- "Show me a Terraform module for AWS VPC"
- "What's the difference between Docker CMD and ENTRYPOINT?"

### Programming
- "Explain Python decorators with examples"
- "How do I use goroutines in Go?"
- "Show me Bash array manipulation"

### Cross-Domain
- "Deploy a Python app on Kubernetes"
- "Write a Bash script to automate Terraform"
- "Compare async patterns in Python and Go"

## Model Recommendations

### Chat & Docs
- **llama3.1:8b** - General purpose (~5GB VRAM)
- **mistral:7b** - Fast responses (~5GB VRAM)

### Coding
- **qwen2.5-coder:7b** - Quick edits (~5GB VRAM)
- **qwen2.5-coder:32b** - Complex refactoring (~19GB VRAM)

## File Locations

```
ai-rag-stack/
├── frontend/src/App.js          # Frontend React app
├── backend/app/main.py          # Backend API
├── scripts/download_docs.sh     # Doc downloader
├── scripts/setup_aider.sh       # Aider setup
├── .aider.conf.yml              # Aider 7B config
├── .aider.32b.conf.yml          # Aider 32B config
├── docker-compose.yml           # Production (Docker Hub)
├── docker-compose.dev.yml       # Development (local builds)
├── data/docs/                   # Downloaded documentation
└── data/custom/                 # Your custom docs
```

## Themes

**Toggle button in header:**
- 🌙 Dark Theme (GitHub-inspired)
- 🎨 Catppuccin Mocha (pastel)

## Troubleshooting

### Services won't start
```bash
docker ps                        # Check running containers
docker compose logs              # Check logs
nvidia-smi                       # Check GPU
```

### Out of memory
```bash
# Use smaller models
docker exec ollama ollama pull llama3.1:8b

# Check VRAM usage
nvidia-smi
```

### Docs not found
```bash
# Re-download
make download-docs

# Re-ingest
make ingest

# Check stats
make stats
```

## Development

```bash
# Frontend rebuild
docker compose -f docker-compose.dev.yml up -d --build frontend

# Backend rebuild
docker compose -f docker-compose.dev.yml up -d --build backend

# Watch logs
make logs-backend
make logs-ollama
```

## Aider Commands

```bash
# In Aider session:
/add <file>         # Add file to context
/drop <file>        # Remove file from context
/diff               # Show pending changes
/undo               # Undo last change
/commit             # Manual commit
/help               # Show all commands
/exit               # Exit Aider
```

## Resources

- **README.md** - Full project documentation
- **AIDER_QUICKSTART.md** - Aider usage guide
- **DOCUMENTATION_GUIDE.md** - Documentation management
- **GitHub**: https://github.com/jconover/ai-rag-stack
- **Docker Hub**:
  - https://hub.docker.com/r/jconover/ai-rag-backend
  - https://hub.docker.com/r/jconover/ai-rag-frontend

---

**Quick Help**: `make help` | **Full Docs**: `cat README.md`
