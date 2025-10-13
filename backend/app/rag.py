"""RAG (Retrieval-Augmented Generation) pipeline"""
from typing import List, Optional, Dict, Any
import ollama
from langchain.schema import Document

from app.config import settings
from app.vectorstore import vector_store


class RAGPipeline:
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.default_model = settings.ollama_default_model
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            source_type = doc.metadata.get('source_type', 'Unknown')
            content = doc.page_content.strip()
            
            context_parts.append(
                f"[Source {i} - {source_type}]\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt with context for the LLM"""
        if context:
            prompt = f"""You are an expert DevOps engineer assistant. Use the provided documentation context to answer the user's question accurately and concisely.

Context from documentation:
{context}

User Question: {query}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't fully answer the question, use your general knowledge but mention this
- Provide code examples when relevant
- Be concise but thorough
- If you're unsure, say so

Answer:"""
        else:
            prompt = f"""You are an expert DevOps engineer assistant. Answer the following question:

{query}

Provide a helpful, accurate, and concise response."""
        
        return prompt
    
    async def generate_response(
        self,
        query: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_rag: bool = True,
    ) -> Dict[str, Any]:
        """Generate response using RAG pipeline"""
        
        model = model or self.default_model
        context_docs = []
        context_str = ""
        
        # Retrieve relevant context if RAG is enabled
        if use_rag:
            try:
                context_docs = vector_store.search(query, top_k=settings.top_k_results)
                context_str = self._format_context(context_docs)
            except Exception as e:
                print(f"Error retrieving context: {e}")
        
        # Build prompt
        prompt = self._build_prompt(query, context_str)
        
        # Generate response using Ollama
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                }
            )
            
            answer = response['message']['content']
            
            # Format sources
            sources = []
            if context_docs:
                for doc in context_docs:
                    sources.append({
                        'source': doc.metadata.get('source', 'Unknown'),
                        'source_type': doc.metadata.get('source_type', 'Unknown'),
                        'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                    })
            
            return {
                'response': answer,
                'model': model,
                'context_used': bool(context_docs),
                'sources': sources if sources else None,
            }
        
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models"""
        try:
            models = ollama.list()
            return models.get('models', [])
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def is_ollama_connected(self) -> bool:
        """Check if Ollama is accessible"""
        try:
            ollama.list()
            return True
        except:
            return False


# Singleton instance
rag_pipeline = RAGPipeline()
