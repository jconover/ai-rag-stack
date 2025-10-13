# Setup Guide - DevOps AI Assistant

Complete setup instructions for getting your local LLM RAG system running.

## Prerequisites

1. **Docker & Docker Compose**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Add your user to docker group
   sudo usermod -aG docker $USER
   newgrp docker
   ```

2. **NVIDIA Container Toolkit** (for GPU support)

   **For Ubuntu 22.04, 24.04, 25.04:**
   ```bash
   # Install nvidia-container-toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit

   # Configure Docker to use NVIDIA runtime
   sudo nvidia-ctk runtime configure --runtime=docker

   # Restart Docker
   sudo systemctl restart docker

   # Test GPU access
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
   ```

   **If already installed (check first):**
   ```bash
   # Verify GPU access works
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

   # If this works, you're all set!
   ```

3. **System Resources**
   - Ensure you have at least 50GB free disk space
   - Close unnecessary applications to free up RAM

## Quick Start

1. **Clone and Setup**
   ```bash
    cd ai-rag-stack
   make setup
   ```

2. **Configure Environment** (optional)
   ```bash
   nano .env
   # Adjust settings if needed
   ```

3. **Start All Services**
   ```bash
   make start
   ```

4. **Pull an LLM Model**
   ```bash
   # Pull default model (8B parameters, ~4.7GB)
   make pull-model
   
   # OR pull CodeLlama for better code generation
   make pull-codellama
   
   # Check available models
   make list-models
   ```

5. **Download and Index Documentation**
   ```bash
   # This will take 10-30 minutes depending on your internet speed
   make download-docs
   
   # Ingest documentation into vector database
   make ingest
   ```

6. **Access the UI**
   - Open browser: http://localhost:3000
   - API docs: http://localhost:8000/docs

## Step-by-Step Setup

### 1. Environment Configuration

The `.env` file controls all configuration:

```bash
# Ollama Configuration
OLLAMA_DEFAULT_MODEL=llama3.1:8b    # Change to your preferred model

# RAG Configuration
CHUNK_SIZE=1000                      # Increase for more context per chunk
CHUNK_OVERLAP=200                    # Overlap between chunks
TOP_K_RESULTS=5                      # Number of docs to retrieve

# If you have more RAM, increase these
CONTEXT_WINDOW=4096                  # Increase to 8192 for larger context
```

### 2. Choosing the Right Model

For your RTX 3090 (24GB VRAM):

| Model | VRAM | Speed | Best For |
|-------|------|-------|----------|
| llama3.1:8b | ~5GB | Fast | General questions |
| codellama:13b | ~8GB | Medium | Code generation |
| mistral:7b | ~4GB | Very Fast | Quick responses |
| llama3.1:70b-q4 | ~20GB | Slow | Best quality |
| deepseek-coder:33b | ~18GB | Slow | Advanced coding |

**Recommendation**: Start with `llama3.1:8b` for testing, then try `codellama:13b` for better code assistance.

```bash
# Pull multiple models to compare
docker exec ollama ollama pull llama3.1:8b
docker exec ollama ollama pull codellama:13b
docker exec ollama ollama pull mistral:7b
```

### 3. Documentation Sources

The `scripts/download_docs.sh` downloads:
- Kubernetes official documentation
- Terraform documentation
- Docker documentation
- Ansible documentation
- Prometheus documentation

**Add Custom Documentation**:
```bash
# Add your own markdown/text files
cp -r /path/to/your/docs/* data/custom/

# Re-run ingestion
make ingest
```

### 4. Monitoring Services

```bash
# Check if all services are healthy
make health

# View all logs
make logs

# View specific service logs
make logs-backend
make logs-ollama

# Check vector database stats
make stats
```

Expected health output:
```json
{
    "status": "healthy",
    "ollama_connected": true,
    "qdrant_connected": true,
    "redis_connected": true
}
```

### 5. Testing the System

```bash
# Test via API
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I create a Kubernetes deployment?",
    "model": "llama3.1:8b",
    "use_rag": true
  }'
```

## Troubleshooting

### Issue: Ollama won't start or can't access GPU

```bash
# Check GPU is detected
nvidia-smi

# Check Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check Ollama logs
docker logs ollama

# Restart Ollama
docker restart ollama
```

### Issue: Out of Memory

1. **Reduce model size**: Use smaller model (7B-8B instead of 13B+)
2. **Reduce context window**: Set `CONTEXT_WINDOW=2048` in `.env`
3. **Reduce concurrent models**: Set `OLLAMA_MAX_LOADED_MODELS=1` in docker-compose.yml

### Issue: Slow responses

1. **GPU not being used**: Check `nvidia-smi` while generating response
2. **Use faster model**: Switch to `mistral:7b`
3. **Reduce Top K**: Set `TOP_K_RESULTS=3` in `.env`
4. **Lower temperature**: Set temperature to 0.3-0.5 in requests

### Issue: No documents found / Empty responses

```bash
# Check if documents were downloaded
ls -lh data/docs/

# Check vector database stats
make stats

# Re-run ingestion
make ingest
```

### Issue: Services won't start

```bash
# Check Docker resources
docker system df

# Check for port conflicts
sudo lsof -i :8000
sudo lsof -i :3000
sudo lsof -i :11434

# Clean and restart
make clean
make start
```

## Performance Tuning

### For Maximum Speed

```bash
# Use faster model
OLLAMA_DEFAULT_MODEL=mistral:7b

# Reduce context
TOP_K_RESULTS=3
CONTEXT_WINDOW=2048

# Increase CPU threads (match your CPU cores)
OLLAMA_NUM_THREAD=16
```

### For Maximum Quality

```bash
# Use larger model
docker exec ollama ollama pull llama3.1:70b-q4

# Increase context
TOP_K_RESULTS=8
CONTEXT_WINDOW=8192
CHUNK_SIZE=1500
```

### For Coding Tasks

```bash
# Use code-specialized model
docker exec ollama ollama pull codellama:13b
# or
docker exec ollama ollama pull deepseek-coder:33b

# In UI, select the code model from dropdown
```

## Updating

```bash
# Pull latest images
docker compose pull

# Restart services
make restart

# Update a specific model
docker exec ollama ollama pull llama3.1:8b
```

## Backup

```bash
# Backup vector database
docker exec qdrant tar czf - /qdrant/storage > qdrant-backup.tar.gz

# Backup custom docs
tar czf custom-docs-backup.tar.gz data/custom/
```

## Next Steps

1. **Test with sample questions** - Try DevOps-related queries
2. **Add your own documentation** - Copy to `data/custom/`
3. **Experiment with models** - Try different models for different tasks
4. **Integrate with tools** - Use the API in your scripts/CI/CD
5. **Monitor performance** - Use `nvidia-smi` and `docker stats`

## Support

- Check logs: `make logs`
- Health check: `make health`
- Stats: `make stats`
- GitHub issues: [Your repo URL]

## Useful Commands Reference

```bash
make help          # Show all commands
make start         # Start all services
make stop          # Stop all services
make restart       # Restart services
make logs          # View all logs
make pull-model    # Pull default model
make list-models   # List available models
make ingest        # Ingest documentation
make health        # Check service health
make stats         # Show database stats
make clean         # Clean containers
```
