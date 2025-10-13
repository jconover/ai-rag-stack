# System Restart Guide

Quick reference for getting your AI DevOps Assistant back up and running after a system restart, reboot, or shutdown.

## Table of Contents
- [Quick Start (TL;DR)](#quick-start-tldr)
- [Full Restart Procedure](#full-restart-procedure)
- [Verification Steps](#verification-steps)
- [Troubleshooting Common Issues](#troubleshooting-common-issues)
- [Performance Checks](#performance-checks)
- [Stopping Services](#stopping-services)

---

## Quick Start (TL;DR)

**Everything in 3 commands:**

```bash
 cd ai-rag-stack
make start
make list-models  # Verify model is available
```

**Access the UI:** http://localhost:3000

That's it! Everything should just work.

---

## Full Restart Procedure

### Step 1: Navigate to Project Directory

```bash
 cd ai-rag-stack
```

### Step 2: Verify Docker is Running

```bash
# Check Docker service status
docker ps

# If Docker isn't running, start it:
sudo systemctl start docker

# Enable Docker to start on boot (one-time setup):
sudo systemctl enable docker
```

### Step 3: Start All Services

```bash
make start
```

**What this does:**
- Starts Ollama (LLM engine)
- Starts Qdrant (vector database)
- Starts Redis (conversation memory)
- Starts Backend API (FastAPI)
- Starts Frontend UI (React)
- Waits for services to be ready (~10 seconds)

**Expected output:**
```
Starting all services...
docker compose up -d
Container redis  Creating
Container qdrant  Creating
Container ollama  Creating
Container redis  Created
Container qdrant  Created
Container ollama  Created
Container rag-backend  Creating
Container rag-backend  Created
Container rag-frontend  Creating
Container rag-frontend  Created
...
Services started!
Frontend: http://localhost:3000
Backend API: http://localhost:8000
API Docs: http://localhost:8000/docs
```

### Step 4: Wait for Services to Initialize

The containers start quickly, but they need a few seconds to fully initialize:

```bash
# Wait 15-20 seconds, then check status
sleep 15
docker ps
```

**All 5 containers should be running:**
- `ollama`
- `qdrant`
- `redis`
- `rag-backend`
- `rag-frontend`

### Step 5: Verify Everything is Working

```bash
# Check service health
make health
```

**Expected output:**
```json
{
    "status": "healthy",
    "ollama_connected": true,
    "qdrant_connected": true,
    "redis_connected": true
}
```

### Step 6: Verify Models are Available

```bash
# List loaded models
make list-models
```

**Expected output:**
```
NAME           ID              SIZE      MODIFIED
llama3.1:8b    46e0c10c039e    4.9 GB    X days ago
```

If no models are listed, see [Troubleshooting](#troubleshooting-common-issues).

### Step 7: Access the Application

**Open your browser:**
- Web UI: http://localhost:3000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/health

**Try a test query in the UI:**
- "What is Kubernetes?"
- "How do I create a Docker container?"

---

## Verification Steps

### Quick Verification

```bash
# All-in-one verification script
bash scripts/verify_setup.sh
```

This checks:
- Docker is running
- GPU is accessible
- All ports are available
- Sufficient disk space
- Sufficient RAM

### Manual Verification

#### 1. Check All Containers are Running

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

**Expected:**
```
NAMES          STATUS
rag-frontend   Up X minutes (healthy)
rag-backend    Up X minutes (healthy)
ollama         Up X minutes (healthy)
qdrant         Up X minutes (healthy)
redis          Up X minutes (healthy)
```

#### 2. Check Service Health

```bash
make health
```

All three connections should be `true`:
- `ollama_connected: true`
- `qdrant_connected: true`
- `redis_connected: true`

#### 3. Check Vector Database

```bash
make stats
```

**Expected:**
```json
{
    "collection_name": "devops_docs",
    "vectors_count": 0,
    "indexed_documents": 55908
}
```

**Note:** `indexed_documents` should be 55,908 (or whatever you indexed).

#### 4. Check GPU Access

```bash
# View GPU status
nvidia-smi

# Check Ollama can see GPU
docker logs ollama | grep GPU
```

**Expected in logs:**
```
msg="inference compute" name=CUDA0 description="NVIDIA GeForce RTX 3090"
```

#### 5. Test API Endpoint

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is Docker?",
    "model": "llama3.1:8b",
    "use_rag": false
  }' | python3 -m json.tool | head -20
```

Should return a JSON response with an answer.

---

## Troubleshooting Common Issues

### Issue 1: Containers Won't Start

**Symptoms:**
- `make start` fails
- Containers exit immediately
- Port conflicts

**Solutions:**

```bash
# Check what's using the ports
sudo lsof -i :3000
sudo lsof -i :8000
sudo lsof -i :6333
sudo lsof -i :11434

# If something is using them, kill it:
sudo kill -9 <PID>

# Or restart Docker entirely:
sudo systemctl restart docker

# Then try again:
make start
```

### Issue 2: No Models Available

**Symptoms:**
- `make list-models` shows nothing
- UI shows "No models available"

**Solution:**

```bash
# Pull the model again
make pull-model

# Or manually:
docker exec ollama ollama pull llama3.1:8b

# Check it downloaded:
make list-models
```

### Issue 3: Services Show "Unhealthy"

**Symptoms:**
- `docker ps` shows "(unhealthy)" status
- Health check fails

**Solutions:**

```bash
# Check logs to see what's wrong
make logs

# Or check specific service:
docker logs ollama
docker logs qdrant
docker logs redis
docker logs rag-backend

# Try restarting:
make restart

# Wait a minute for health checks:
sleep 60
docker ps
```

### Issue 4: GPU Not Detected

**Symptoms:**
- Ollama logs don't show GPU
- Inference is very slow
- nvidia-smi shows no Docker processes

**Solutions:**

```bash
# 1. Check NVIDIA driver is working
nvidia-smi

# 2. Check Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# 3. Restart Docker with GPU support
sudo systemctl restart docker

# 4. Restart services
make restart

# 5. Check Ollama logs again
docker logs ollama 2>&1 | grep GPU
```

### Issue 5: Frontend Won't Load

**Symptoms:**
- http://localhost:3000 shows error
- Page is blank
- Connection refused

**Solutions:**

```bash
# Check frontend container
docker logs rag-frontend

# Check if port 3000 is in use
sudo lsof -i :3000

# Restart just the frontend
docker restart rag-frontend

# Wait a few seconds
sleep 5

# Try accessing again
curl http://localhost:3000
```

### Issue 6: Backend API Errors

**Symptoms:**
- API returns 500 errors
- Connection errors in frontend
- "Service unavailable"

**Solutions:**

```bash
# Check backend logs
docker logs rag-backend

# Common issues:
# - Can't connect to Ollama
# - Can't connect to Qdrant
# - Can't connect to Redis

# Verify all services are up:
docker ps

# Check backend can reach other services:
docker exec rag-backend curl -f http://ollama:11434
docker exec rag-backend curl -f http://qdrant:6333/health

# Restart backend:
docker restart rag-backend
```

### Issue 7: Slow Response Times

**Symptoms:**
- Queries take >30 seconds
- GPU not being utilized
- System feels sluggish

**Solutions:**

```bash
# 1. Check GPU usage during query
watch -n 1 nvidia-smi
# GPU should show 70-90% utilization during generation

# 2. Check system resources
htop

# 3. Check if model needs to be loaded (first query)
# First query after restart is slower (~10 seconds)
# Subsequent queries should be 2-5 seconds

# 4. Make sure GPU is being used
docker logs ollama 2>&1 | grep "inference compute"

# 5. Restart Ollama to reload model
docker restart ollama
```

### Issue 8: Out of Memory

**Symptoms:**
- Containers crash
- System becomes unresponsive
- OOM (Out of Memory) errors in logs

**Solutions:**

```bash
# Check memory usage
free -h

# Check Docker memory
docker stats

# If memory is low:
# 1. Close other applications
# 2. Use a smaller model:
docker exec ollama ollama pull mistral:7b

# 3. Restart with fresh memory:
make stop
make start
```

---

## Performance Checks

### Check Response Time

```bash
# Time a simple query
time curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Kubernetes?", "model": "llama3.1:8b"}' \
  -o /dev/null -s
```

**Expected:** 2-5 seconds after model is loaded

### Monitor GPU During Query

```bash
# In one terminal, start monitoring:
watch -n 1 nvidia-smi

# In another terminal, send query:
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain Docker containers", "model": "llama3.1:8b"}'
```

**Expected GPU usage during generation:**
- Utilization: 70-90%
- Memory: 5-6GB (for llama3.1:8b)
- Power: 200-300W

### Check Container Resource Usage

```bash
docker stats
```

**Expected:**
- `ollama`: 5-10GB RAM, high CPU during inference
- `rag-backend`: 1-2GB RAM
- `qdrant`: 500MB-1GB RAM
- `redis`: 50-100MB RAM
- `rag-frontend`: 50MB RAM

---

## Stopping Services

### Stop All Services (Keep Data)

```bash
make stop
```

This stops containers but preserves:
- Downloaded models (in `ollama_data` volume)
- Indexed documents (in `qdrant_data` volume)
- Conversation history (in `redis_data` volume)

### Stop and Remove Everything (Keep Data)

```bash
make clean
```

Same as `make stop` but also removes containers.

### Stop and Delete ALL Data

**⚠️ WARNING: This deletes all models and indexed documents!**

```bash
make clean-all
```

This removes:
- All containers
- All Docker volumes (models, vector DB, Redis data)
- Downloaded documentation files

**You'll need to:**
1. Re-pull models: `make pull-model`
2. Re-download docs: `make download-docs`
3. Re-index: `make ingest`

---

## Common Commands Quick Reference

```bash
# Starting
make start          # Start all services
make restart        # Restart all services

# Checking Status
make health         # Check service health
make stats          # Vector DB statistics
make list-models    # List available models
docker ps           # See running containers

# Viewing Logs
make logs           # All service logs
make logs-ollama    # Ollama logs only
make logs-backend   # Backend logs only
docker logs <name>  # Specific container

# GPU Monitoring
nvidia-smi                    # Current GPU status
watch -n 1 nvidia-smi        # Real-time monitoring
docker logs ollama | grep GPU # Check GPU detection

# Stopping
make stop           # Stop services (keep data)
make clean          # Stop and remove containers (keep data)
make clean-all      # Stop and delete everything

# Help
make help           # Show all available commands
```

---

## Startup Time Expectations

After running `make start`:

| Service | Startup Time | Ready When |
|---------|--------------|------------|
| Redis | 2-3 seconds | Immediate |
| Qdrant | 3-5 seconds | Listening on 6333 |
| Ollama | 5-10 seconds | GPU detected in logs |
| Backend | 10-15 seconds | Health endpoint responds |
| Frontend | 15-20 seconds | UI loads in browser |

**Total time to fully operational:** ~20-30 seconds

**First query after restart:** 8-15 seconds (model loading)
**Subsequent queries:** 2-5 seconds

---

## Auto-Start on Boot (Optional)

If you want services to start automatically when you boot your computer:

### Option 1: Docker Compose (Recommended)

Already configured! The `restart: unless-stopped` in docker-compose.yml means containers will auto-start.

**Just ensure Docker starts on boot:**
```bash
sudo systemctl enable docker
```

### Option 2: Systemd Service

Create a systemd service to manage the stack:

```bash
# Create service file
sudo nano /etc/systemd/system/ai-rag-stack.service
```

```ini
[Unit]
Description=AI DevOps Assistant RAG Stack
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/justin/Code/ai-rag-stack
ExecStart=/usr/bin/make start
ExecStop=/usr/bin/make stop
User=justin

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable ai-rag-stack.service
sudo systemctl start ai-rag-stack.service

# Check status
sudo systemctl status ai-rag-stack.service
```

### Option 3: Cron @reboot

```bash
# Edit crontab
crontab -e

# Add this line:
@reboot sleep 30 &&  cd ai-rag-stack && make start
```

---

## Post-Restart Checklist

After system restart, verify:

```bash
✅ Docker is running: docker ps
✅ All 5 containers up: docker ps | grep -c "Up" (should be 5)
✅ Services healthy: make health
✅ Models available: make list-models
✅ GPU detected: docker logs ollama | grep GPU
✅ UI accessible: curl http://localhost:3000
✅ API working: make test
✅ Vector DB has data: make stats (indexed_documents > 0)
```

If all checks pass, you're good to go! 🚀

---

## FAQ

**Q: Do I need to re-pull models after restart?**
A: No. Models are stored in Docker volumes and persist across restarts.

**Q: Do I need to re-index documentation?**
A: No. The vector database persists in the `qdrant_data` volume.

**Q: How long does it take to start everything?**
A: About 20-30 seconds from `make start` to fully operational.

**Q: Will my conversation history be saved?**
A: Yes, for 24 hours. Conversations are stored in Redis with TTL.

**Q: What if I upgrade Docker or the OS?**
A: Your data should be safe. Volumes persist through upgrades. But backup first: `docker exec qdrant tar czf - /qdrant/storage > backup.tar.gz`

**Q: Can I update the models without losing data?**
A: Yes. Pull new models with `docker exec ollama ollama pull <model-name>`. Old models stay until you delete them.

**Q: What's using my disk space?**
A: Check Docker volumes: `docker system df -v`

**Q: How do I free up disk space?**
A: Remove unused images: `docker system prune -a`

---

## Summary

**After any restart, just run:**

```bash
 cd ai-rag-stack
make start
```

**Then verify:**

```bash
make health
make list-models
```

**Access:** http://localhost:3000

**That's it!** Your AI DevOps Assistant is back online. 🎉

---

## Quick Troubleshooting Flowchart

```
System Restarted
      |
      v
Run: make start
      |
      ├─> Services start ──> SUCCESS ✓
      |
      └─> Services fail
            |
            ├─> Port conflict? ──> Kill process on port
            |
            ├─> Docker not running? ──> sudo systemctl start docker
            |
            ├─> No models? ──> make pull-model
            |
            ├─> Unhealthy status? ──> Check logs: make logs
            |
            ├─> GPU not detected? ──> Restart Docker, check nvidia-smi
            |
            └─> Other issues? ──> make clean && make start
```

---

**Need more help?** Check the main documentation:
- [SETUP.md](SETUP.md) - Detailed setup and troubleshooting
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - File organization

**Still stuck?** Run the verification script:
```bash
bash scripts/verify_setup.sh
```
