"""Qdrant vector store management"""
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from app.config import settings


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        # Warmup the embedding model to eliminate first-query cold-start latency
        self.embeddings.embed_query("warmup")
        self.collection_name = settings.qdrant_collection_name
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,  # all-MiniLM-L6-v2 embedding dimension
                    distance=Distance.COSINE
                )
            )
    
    def search(self, query: str, top_k: int = None) -> List[Document]:
        """Search for relevant documents"""
        if top_k is None:
            top_k = settings.top_k_results
        
        vectorstore = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )
        
        results = vectorstore.similarity_search(query, k=top_k)
        return results
    
    def get_stats(self) -> dict:
        """Get collection statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count or 0,
                "points_count": collection_info.points_count or 0,
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "vectors_count": 0,
                "points_count": 0,
                "error": str(e)
            }
    
    def is_connected(self) -> bool:
        """Check if Qdrant is connected"""
        try:
            self.client.get_collections()
            return True
        except:
            return False


# Singleton instance
vector_store = VectorStore()
