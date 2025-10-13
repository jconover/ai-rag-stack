# 🚀 START HERE - DevOps AI Assistant

Welcome to your AI-powered DevOps assistant! This guide will get you running in **5 minutes**.

## ✅ Your System is Ready!

Good news: Your Ubuntu 25.04 system with RTX 3090 is **already verified and ready**.

## 📋 Quick Start (3 Commands)

```bash
 cd ai-rag-stack

# 1. Setup environment
make setup

# 2. Start all services
make start

# 3. Pull a model
make pull-model
```

**That's it!** Open http://localhost:3000 in your browser.

## 🎯 First Steps After Starting

### Try Without Documentation First
Even without indexing documentation, you can test the system:
1. Open http://localhost:3000
2. Ask: "What is Kubernetes?"
3. The LLM will respond using its trained knowledge

### Add DevOps Documentation (Optional, takes 20-30 minutes)
```bash
# Download K8s, Terraform, Docker, Ansible docs
make download-docs

# Index them into the vector database
make ingest
```

Now queries will include relevant documentation snippets!

## 📚 Documentation Guide

Start with these based on what you need:

### Just Want to Get Started?
→ Read: **QUICKSTART_UBUNTU_25.04.md** (5 min read)

### Want to Understand the System?
→ Read: **README.md** then **ARCHITECTURE.md** (15 min read)

### Need Setup Help?
→ Read: **SETUP.md** (detailed troubleshooting)

### Curious About Your System?
→ Read: **YOUR_SYSTEM_STATUS.md** (your hardware status)

### Want to Contribute?
→ Read: **CONTRIBUTING.md**

## 🎮 Available Commands

```bash
make help          # Show all commands
make verify        # Verify system requirements
make start         # Start all services
make stop          # Stop services
make logs          # View logs
make health        # Check service health
make stats         # Database statistics
make test          # Test API endpoints
```

## 🤖 Model Recommendations

With your 24GB RTX 3090, try these:

**Start Here (Fast & Good)**
```bash
make pull-model  # Pulls llama3.1:8b (~5GB)
```

**For Better Code**
```bash
make pull-codellama  # Pulls codellama:13b (~8GB)
```

**Load Multiple** (you have room!)
```bash
docker exec ollama ollama pull llama3.1:8b
docker exec ollama ollama pull codellama:13b
docker exec ollama ollama pull mistral:7b
```

## 🌐 Service URLs

Once running:
- **Web UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/api/health
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## ⚡ What to Expect

### First Query
- Model loads: 5-10 seconds (first time only)
- Response: 3-5 seconds
- **Total: 8-15 seconds**

### Subsequent Queries
- Response: 2-5 seconds (model stays loaded)

### GPU Usage
- Idle: ~0.5GB VRAM
- During generation: 5-8GB VRAM (depending on model)
- Utilization: 70-90% while generating

## 🎓 Example Queries to Try

```
"How do I create a Kubernetes deployment?"
"Explain Terraform state management"
"What's the difference between CMD and ENTRYPOINT in Docker?"
"Show me an Ansible playbook example"
"How do I monitor with Prometheus?"
```

## 🔍 Verify Everything is Working

```bash
# Check all services are healthy
make health

# Should show:
# {
#   "status": "healthy",
#   "ollama_connected": true,
#   "qdrant_connected": true,
#   "redis_connected": true
# }
```

## 🐛 Quick Troubleshooting

### Services Won't Start
```bash
make logs  # Check what went wrong
make restart  # Try restarting
```

### Can't Access GPU
```bash
nvidia-smi  # Should show your RTX 3090
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### No Response from AI
```bash
make logs-ollama  # Check Ollama logs
make list-models  # Verify model is pulled
```

## 📊 Monitor Your System

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# View all service logs
make logs

# Check vector database
make stats
```

## 🎯 Success Checklist

- [ ] Services started: `make start`
- [ ] Health check passes: `make health`
- [ ] Model pulled: `make list-models` shows at least one
- [ ] UI loads: http://localhost:3000
- [ ] Test query works
- [ ] GPU active during generation: `nvidia-smi`

## 💡 Tips for Best Experience

1. **Start with small model** (llama3.1:8b) to test everything works
2. **Pull multiple models** to compare performance
3. **Add custom docs** to `data/custom/` for your specific needs
4. **Monitor GPU** with `nvidia-smi` to see it working
5. **Check logs** if anything seems slow or broken

## 🚦 Current Status

Your system verification showed:
- ✅ Docker & GPU access working
- ✅ 24GB VRAM available
- ✅ 128GB RAM
- ✅ 3.3TB disk space
- ✅ All ports available

**You're ready to go!**

## 📖 Next Steps After Getting It Running

1. **Test basic functionality** - Ask a few questions
2. **Download documentation** - `make download-docs` (optional)
3. **Ingest docs** - `make ingest` (optional)
4. **Try different models** - Compare responses
5. **Add custom docs** - Put your own docs in `data/custom/`
6. **Integrate with tools** - Use the API in your workflows

## 🎉 Portfolio Highlights

This project demonstrates:
- ✅ Docker & containerization
- ✅ Microservices architecture
- ✅ AI/ML integration (LLMs, RAG, vectors)
- ✅ Full-stack development (FastAPI + React)
- ✅ GPU optimization
- ✅ DevOps best practices
- ✅ Documentation & testing

## 🆘 Need Help?

1. Check the logs: `make logs`
2. Run verification: `make verify`
3. Read SETUP.md for detailed troubleshooting
4. Check YOUR_SYSTEM_STATUS.md for your configuration

---

## 🏁 Ready? Let's Go!

```bash
 cd ai-rag-stack
make setup && make start
```

Then open http://localhost:3000 and start asking questions!

**Happy building!** 🚀
