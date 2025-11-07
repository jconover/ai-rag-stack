.PHONY: help verify setup start start-dev stop restart logs clean pull-model ingest health publish aider aider-32b aider-deepseek aider-deepseek-33b setup-aider setup-aider-deepseek update-docs

help:
	@echo "DevOps AI Assistant - Available Commands"
	@echo "========================================"
	@echo "verify         - Verify system requirements (Docker, GPU, etc.)"
	@echo "setup          - Initial setup (copy .env, create directories)"
	@echo "start          - Start all services (uses Docker Hub images)"
	@echo "start-dev      - Start services in dev mode (builds locally)"
	@echo "stop           - Stop all services"
	@echo "restart        - Restart all services"
	@echo "logs           - View logs from all services"
	@echo "logs-backend   - View backend logs"
	@echo "logs-ollama    - View Ollama logs"
	@echo "pull-model     - Pull default Ollama model (llama3.1:8b)"
	@echo "pull-model MODEL=<name> - Pull specific Ollama model"
	@echo "pull-codellama - Pull CodeLlama model"
	@echo "list-models    - List available Ollama models"
	@echo "ingest         - Download and ingest DevOps documentation"
	@echo "download-docs  - Download documentation only"
	@echo "update-docs    - Update existing documentation to latest versions"
	@echo "health         - Check service health"
	@echo "stats          - Show vector database statistics"
	@echo "test           - Test API endpoints"
	@echo "publish        - Build and push images to Docker Hub"
	@echo "setup-aider    - Setup Aider coding assistant with Qwen2.5-Coder"
	@echo "setup-aider-deepseek - Setup Aider with DeepSeek Coder models"
	@echo "aider          - Start Aider with qwen2.5-coder:7b (fast)"
	@echo "aider-32b      - Start Aider with qwen2.5-coder:32b (powerful)"
	@echo "aider-deepseek - Start Aider with deepseek-coder:6.7b"
	@echo "aider-deepseek-33b - Start Aider with deepseek-coder:33b (most powerful)"
	@echo "clean          - Clean up containers and volumes"
	@echo "clean-all      - Clean everything including data"

verify:
	@bash scripts/verify_setup.sh

setup:
	@echo "Setting up environment..."
	@cp --update=none .env.example .env || true
	@mkdir -p data/docs data/custom
	@echo "Setup complete! Edit .env if needed, then run 'make start'"

start:
	@echo "Starting all services (using Docker Hub images)..."
	docker compose pull
	docker compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Services started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

start-dev:
	@echo "Starting all services in DEV mode (building locally)..."
	docker compose -f docker-compose.dev.yml up -d --build
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Services started in DEV mode!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

stop:
	docker compose down
	docker compose -f docker-compose.dev.yml down

restart:
	docker compose restart

logs:
	docker compose logs -f

logs-backend:
	docker compose logs -f backend

logs-ollama:
	docker compose logs -f ollama

pull-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "Pulling default llama3.1:8b model..."; \
		docker exec ollama ollama pull llama3.1:8b; \
	else \
		echo "Pulling $(MODEL) model..."; \
		docker exec ollama ollama pull $(MODEL); \
	fi

pull-codellama:
	@echo "Pulling codellama:13b model..."
	docker exec ollama ollama pull codellama:13b

list-models:
	docker exec ollama ollama list

ingest: download-docs
	@echo "Ingesting documentation into vector database..."
	docker exec rag-backend python /scripts/ingest_docs.py

download-docs:
	@echo "Downloading DevOps documentation..."
	bash scripts/download_docs.sh data/docs

update-docs:
	@echo "Updating existing documentation..."
	@bash scripts/update_docs.sh data/docs && \
		echo "" && \
		echo "📚 Updates detected! Re-ingesting documentation..." && \
		$(MAKE) ingest || \
		echo "✓ No updates found. Documentation is current."

health:
	@curl -s http://localhost:8000/api/health | python3 -m json.tool

stats:
	@curl -s http://localhost:8000/api/stats | python3 -m json.tool

test:
	@bash scripts/test_api.sh

publish:
	@echo "Building and pushing images to Docker Hub..."
	@bash scripts/push_to_dockerhub.sh

setup-aider:
	@bash scripts/setup_aider.sh

aider:
	@echo "Starting Aider with qwen2.5-coder:7b (fast)..."
	@OLLAMA_API_BASE=http://localhost:11434 aider --config .aider.conf.yml

aider-32b:
	@echo "Starting Aider with qwen2.5-coder:32b (powerful)..."
	@OLLAMA_API_BASE=http://localhost:11434 aider --config .aider.32b.conf.yml

setup-aider-deepseek:
	@echo "Setting up Aider with DeepSeek Coder models..."
	@echo "Installing Aider (if not already installed)..."
	@pip install --upgrade aider-chat || echo "Aider already installed or pip not available"
	@echo ""
	@echo "Pulling DeepSeek Coder models..."
	@echo "1/3: Pulling deepseek-coder:1.3b (lightweight, ~800MB)..."
	@docker exec ollama ollama pull deepseek-coder:1.3b
	@echo ""
	@echo "2/3: Pulling deepseek-coder:6.7b (recommended, ~3.8GB)..."
	@docker exec ollama ollama pull deepseek-coder:6.7b
	@echo ""
	@echo "3/3: Pulling deepseek-coder:33b (powerful, ~18GB)..."
	@docker exec ollama ollama pull deepseek-coder:33b
	@echo ""
	@echo "✓ DeepSeek Coder setup complete!"
	@echo ""
	@echo "Usage:"
	@echo "  make aider-deepseek       - Start with deepseek-coder:6.7b"
	@echo "  make aider-deepseek-33b   - Start with deepseek-coder:33b (most powerful)"
	@echo "  Or run directly:"
	@echo "  aider --model ollama/deepseek-coder:6.7b"
	@echo "  aider --model ollama/deepseek-coder:33b"

aider-deepseek:
	@echo "Starting Aider with deepseek-coder:6.7b..."
	@OLLAMA_API_BASE=http://localhost:11434 aider --config .aider.deepseek.conf.yml

aider-deepseek-33b:
	@echo "Starting Aider with deepseek-coder:33b (most powerful)..."
	@OLLAMA_API_BASE=http://localhost:11434 aider --config .aider.deepseek-33b.conf.yml

clean:
	docker compose down -v
	docker compose -f docker-compose.dev.yml down -v

clean-all: clean
	@echo "Removing all data..."
	rm -rf data/docs/*
	@echo "Clean complete!"
