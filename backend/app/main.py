"""FastAPI application for DevOps AI Assistant"""
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import redis
import json

from app.config import settings
from app.models import (
    ChatRequest, ChatResponse, HealthResponse, 
    ModelsResponse, ModelInfo, StatsResponse
)
from app.rag import rag_pipeline
from app.vectorstore import vector_store

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
