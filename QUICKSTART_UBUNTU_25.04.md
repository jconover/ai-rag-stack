# Quick Start Guide - Ubuntu 25.04

Your system is **ready to go**! Here's the fastest path to get the AI assistant running.

## Your System Status ✓

- ✅ Ubuntu 25.04 "Plucky Platypus"
- ✅ NVIDIA Driver 570.172.08 with CUDA 12.8
- ✅ RTX 3090 24GB detected and working
- ✅ Docker 28.5.1 with GPU access configured
- ✅ 128GB RAM, 3.3TB disk space

## 5-Minute Setup

### 1. Verify Everything is Ready

```bash
 cd ai-rag-stack
bash scripts/verify_setup.sh
```

You should see all green checkmarks. If not, the script will tell you what to fix.

### 2. Initial Setup

```bash
# Create .env file and directories
make setup
```

### 3. Start All Services

```bash
# Start Ollama, Qdrant, Redis, Backend API, Frontend UI
make start

# Wait ~30 seconds for services to be ready
```

### 4. Pull an LLM Model

**Option A: General Purpose (Recommended First)**
```bash
# Llama 3.1 8B - Fast, general purpose (~5GB VRAM)
make pull-model
```

**Option B: Better for Code**
```bash
# CodeLlama 13B - Better code generation (~8GB VRAM)
make pull-codellama
```

**Option C: Multiple Models (You have 24GB!)**
```bash
# Pull several to compare
docker exec ollama ollama pull llama3.1:8b
docker exec ollama ollama pull codellama:13b
docker exec ollama ollama pull mistral:7b
```

### 5. Download & Index DevOps Documentation

```bash
# Download K8s, Terraform, Docker, Ansible, Prometheus docs
# This takes 10-20 minutes depending on internet speed
make download-docs

# Index documents into vector database
# This takes 5-10 minutes
make ingest
```

### 6. Access the UI

Open your browser to: **http://localhost:3000**

Try asking:
- "How do I create a Kubernetes deployment?"
- "Explain Terraform modules"
- "What's the difference between Docker CMD and ENTRYPOINT?"

## System URLs

- **Web UI**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Monitoring Commands

```bash
# Check if services are healthy
make health

# View all logs
make logs

# View specific service logs
make logs-backend
make logs-ollama

# Check vector database stats
make stats

# List installed models
make list-models
```

## Recommended Model Strategy

With your 24GB VRAM, you can:

1. **Start Simple**: Begin with `llama3.1:8b` (~5GB) for testing
2. **Add Code Model**: Pull `codellama:13b` (~8GB) for coding tasks
3. **Load Both**: With 24GB, you can keep 2-3 models loaded simultaneously

```bash
# Pull multiple models
docker exec ollama ollama pull llama3.1:8b      # General
docker exec ollama ollama pull codellama:13b    # Code
docker exec ollama ollama pull mistral:7b       # Fast responses

# They'll auto-load when selected in the UI
```

## What to Expect

### First Query (Cold Start)
- Model loading: 5-10 seconds
- Response generation: 3-5 seconds
- **Total: 8-15 seconds**

### Subsequent Queries (Warm)
- Response generation: 2-5 seconds
- With RAG context retrieval adds <1 second

### Performance
- **Llama 3.1 8B**: ~40 tokens/second
- **CodeLlama 13B**: ~25 tokens/second
- **Mistral 7B**: ~50 tokens/second

## Troubleshooting

### Services won't start
```bash
# Check what's running
docker ps

# View specific service logs
docker logs ollama
docker logs qdrant
docker logs rag-backend

# Restart everything
make restart
```

### No responses / Empty answers
```bash
# Make sure documents are indexed
make stats

# Should show vectors_count > 0
# If not, run: make ingest
```

### Slow responses
```bash
# Check GPU usage while generating
watch -n 1 nvidia-smi

# GPU should be at 70-90% utilization during generation
```

### Out of memory
```bash
# Use a smaller model
docker exec ollama ollama pull mistral:7b

# Check VRAM usage
nvidia-smi
```

## Next Steps

1. **Test the system** with example queries
2. **Add custom docs** to `data/custom/` for your specific use case
3. **Experiment with models** - switch in the UI dropdown
4. **Integrate with your tools** - use the REST API at localhost:8000

## Common Commands

```bash
make help          # Show all available commands
make start         # Start all services
make stop          # Stop all services
make restart       # Restart services
make logs          # View logs
make health        # Check service health
make stats         # Database statistics
make clean         # Clean up containers
```

## Getting Help

If you run into issues:
1. Run `bash scripts/verify_setup.sh` again
2. Check `make logs` for error messages
3. Ensure ports 3000, 8000, 6333, 11434 are not in use
4. Verify GPU is not being used by other processes: `nvidia-smi`

## Performance Tuning for Your Hardware

Your AMD 9950X + RTX 3090 + 128GB RAM is **excellent** for this workload:

```bash
# Edit docker-compose.yml to optimize for your CPU
OLLAMA_NUM_THREAD=16  # Match your core count

# Try larger models
docker exec ollama ollama pull llama3.1:70b-q4  # Uses ~20GB VRAM
docker exec ollama ollama pull deepseek-coder:33b  # Great for code
```

## Success Indicators

When everything is working:
- ✅ All services show "healthy" in `make health`
- ✅ `make stats` shows vectors_count > 0
- ✅ UI loads at localhost:3000
- ✅ Can select model from dropdown
- ✅ Queries return contextual answers with sources
- ✅ `nvidia-smi` shows GPU activity during generation

**You're ready to build your DevOps AI assistant!** 🚀
