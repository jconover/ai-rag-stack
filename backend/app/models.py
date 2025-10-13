"""Pydantic models for API requests and responses"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message/question")
    model: Optional[str] = Field(None, description="Ollama model to use")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192, description="Maximum tokens to generate")
    use_rag: Optional[bool] = Field(True, description="Whether to use RAG for context")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI assistant response")
    model: str = Field(..., description="Model used for generation")
    context_used: bool = Field(..., description="Whether RAG context was used")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source documents used")
    session_id: str = Field(..., description="Session ID")


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    qdrant_connected: bool
    redis_connected: bool


class ModelInfo(BaseModel):
    name: str
    size: Optional[str] = None
    modified: Optional[str] = None


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class StatsResponse(BaseModel):
    collection_name: str
    vectors_count: int
    indexed_documents: int


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Path to documents to ingest")
    source_name: str = Field(..., description="Name/label for this document source")


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
