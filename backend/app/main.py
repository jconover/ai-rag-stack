"""FastAPI application for DevOps AI Assistant"""
import uuid
import os
import subprocess
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import redis
import json

from app.config import settings
from app.models import (
    ChatRequest, ChatResponse, HealthResponse,
    ModelsResponse, ModelInfo, StatsResponse
)
from app.rag import rag_pipeline
from app.vectorstore import vector_store
from app.templates import get_templates, get_template_by_id, get_categories

# Initialize FastAPI app
app = FastAPI(
    title="DevOps AI Assistant API",
    description="RAG-powered AI assistant for DevOps documentation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client for conversation memory
redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=True
)


def get_conversation_history(session_id: str, limit: int = 5) -> list:
    """Get conversation history from Redis"""
    try:
        history_key = f"chat:{session_id}"
        messages = redis_client.lrange(history_key, -limit, -1)
        return [json.loads(msg) for msg in messages]
    except:
        return []


def save_message(session_id: str, role: str, content: str):
    """Save message to conversation history"""
    try:
        history_key = f"chat:{session_id}"
        message = json.dumps({"role": role, "content": content})
        redis_client.rpush(history_key, message)
        redis_client.expire(history_key, 86400)  # 24 hour expiry
    except Exception as e:
        print(f"Error saving message: {e}")


@app.get("/")
async def root():
    return {
        "message": "DevOps AI Assistant API",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    ollama_connected = rag_pipeline.is_ollama_connected()
    qdrant_connected = vector_store.is_connected()
    
    try:
        redis_client.ping()
        redis_connected = True
    except:
        redis_connected = False
    
    return HealthResponse(
        status="healthy" if all([ollama_connected, qdrant_connected, redis_connected]) else "degraded",
        ollama_connected=ollama_connected,
        qdrant_connected=qdrant_connected,
        redis_connected=redis_connected
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI assistant"""

    # Generate or use provided session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Save user message
    save_message(session_id, "user", request.message)

    try:
        # Generate response
        result = await rag_pipeline.generate_response(
            query=request.message,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            use_rag=request.use_rag,
        )

        # Save assistant response
        save_message(session_id, "assistant", result['response'])

        return ChatResponse(
            response=result['response'],
            model=result['model'],
            context_used=result['context_used'],
            sources=result['sources'],
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses from the AI assistant"""

    # Generate or use provided session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Save user message
    save_message(session_id, "user", request.message)

    async def generate():
        full_response = ""
        try:
            async for chunk in rag_pipeline.generate_response_stream(
                query=request.message,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                use_rag=request.use_rag,
            ):
                # Accumulate response content
                if chunk.get('type') == 'content':
                    full_response += chunk.get('content', '')

                # Add session_id to metadata
                if chunk.get('type') == 'metadata':
                    chunk['session_id'] = session_id

                # Stream as Server-Sent Events format
                yield f"data: {json.dumps(chunk)}\n\n"

            # Save complete assistant response
            if full_response:
                save_message(session_id, "assistant", full_response)

        except Exception as e:
            error_chunk = {
                'type': 'error',
                'error': str(e)
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/models", response_model=ModelsResponse)
async def list_models():
    """List available Ollama models"""
    try:
        models = rag_pipeline.list_models()
        model_infos = [
            ModelInfo(
                name=m.get('name', ''),
                size=str(m.get('size', '')),
                modified=m.get('modified_at', '')
            )
            for m in models
        ]
        return ModelsResponse(models=model_infos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get vector database statistics"""
    try:
        stats = vector_store.get_stats()
        return StatsResponse(
            collection_name=stats['collection_name'],
            vectors_count=stats.get('vectors_count', 0),
            indexed_documents=stats.get('points_count', 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/{session_id}")
async def get_history(session_id: str, limit: int = 10):
    """Get conversation history for a session"""
    history = get_conversation_history(session_id, limit)
    return {"session_id": session_id, "messages": history}


@app.get("/api/templates")
async def list_templates(category: Optional[str] = None):
    """Get prompt templates, optionally filtered by category"""
    templates = get_templates()
    if category:
        templates = [t for t in templates if t['category'] == category]
    return {
        "templates": templates,
        "categories": get_categories()
    }


@app.get("/api/templates/{template_id}")
async def get_template(template_id: str):
    """Get a specific prompt template by ID"""
    template = get_template_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template


@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    auto_ingest: bool = True
):
    """Upload documentation files and optionally trigger ingestion"""

    # Create custom docs directory if it doesn't exist
    custom_docs_dir = Path("/data/custom")
    custom_docs_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files = []
    failed_files = []

    # Save uploaded files
    for file in files:
        try:
            # Validate file type
            if not (file.filename.endswith('.md') or
                    file.filename.endswith('.txt') or
                    file.filename.endswith('.markdown')):
                failed_files.append({
                    "filename": file.filename,
                    "error": "Only .md, .txt, and .markdown files are supported"
                })
                continue

            # Save file
            file_path = custom_docs_dir / file.filename
            content = await file.read()

            with open(file_path, 'wb') as f:
                f.write(content)

            uploaded_files.append({
                "filename": file.filename,
                "size": len(content),
                "path": str(file_path)
            })

        except Exception as e:
            failed_files.append({
                "filename": file.filename,
                "error": str(e)
            })

    result = {
        "uploaded": len(uploaded_files),
        "failed": len(failed_files),
        "files": uploaded_files,
        "errors": failed_files if failed_files else None,
    }

    # Trigger ingestion if requested and files were uploaded
    if auto_ingest and uploaded_files:
        try:
            # Run ingestion script in background
            ingestion_result = subprocess.run(
                ["python", "/scripts/ingest_docs.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            result["ingestion"] = {
                "status": "success" if ingestion_result.returncode == 0 else "failed",
                "output": ingestion_result.stdout,
                "error": ingestion_result.stderr if ingestion_result.returncode != 0 else None
            }
        except subprocess.TimeoutExpired:
            result["ingestion"] = {
                "status": "timeout",
                "error": "Ingestion took longer than 5 minutes"
            }
        except Exception as e:
            result["ingestion"] = {
                "status": "error",
                "error": str(e)
            }

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
