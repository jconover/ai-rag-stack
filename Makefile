.PHONY: help verify setup start stop restart logs clean pull-model ingest health

help:
	@echo "DevOps AI Assistant - Available Commands"
	@echo "========================================"
	@echo "verify         - Verify system requirements (Docker, GPU, etc.)"
	@echo "setup          - Initial setup (copy .env, create directories)"
	@echo "start          - Start all services"
	@echo "stop           - Stop all services"
	@echo "restart        - Restart all services"
	@echo "logs           - View logs from all services"
	@echo "logs-backend   - View backend logs"
	@echo "logs-ollama    - View Ollama logs"
	@echo "pull-model     - Pull default Ollama model (llama3.1:8b)"
	@echo "pull-codellama - Pull CodeLlama model"
	@echo "list-models    - List available Ollama models"
	@echo "ingest         - Download and ingest DevOps documentation"
	@echo "download-docs  - Download documentation only"
	@echo "health         - Check service health"
	@echo "stats          - Show vector database statistics"
	@echo "test           - Test API endpoints"
	@echo "clean          - Clean up containers and volumes"
	@echo "clean-all      - Clean everything including data"

verify:
	@bash scripts/verify_setup.sh

setup:
	@echo "Setting up environment..."
	@cp -n .env.example .env || true
	@mkdir -p data/docs data/custom
	@echo "Setup complete! Edit .env if needed, then run 'make start'"

start:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Services started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

stop:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-ollama:
	docker-compose logs -f ollama

pull-model:
	@echo "Pulling llama3.1:8b model..."
	docker exec ollama ollama pull llama3.1:8b

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

health:
	@curl -s http://localhost:8000/api/health | python3 -m json.tool

stats:
	@curl -s http://localhost:8000/api/stats | python3 -m json.tool

test:
	@bash scripts/test_api.sh

clean:
	docker-compose down -v

clean-all: clean
	@echo "Removing all data..."
	rm -rf data/docs/*
	@echo "Clean complete!"
