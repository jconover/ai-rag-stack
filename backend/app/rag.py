"""RAG (Retrieval-Augmented Generation) pipeline"""
import asyncio
from typing import List, Optional, Dict, Any
import ollama

from app.config import settings
from app.vectorstore import vector_store


class RAGPipeline:
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.default_model = settings.ollama_default_model
    
    def _format_context(self, documents: List) -> str:
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
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt defining the assistant's role and behavior."""
        return """You are an expert DevOps engineer assistant. Your role is to help users with DevOps, infrastructure, and programming questions.

Instructions:
- Answer based primarily on the provided context when available
- If the context doesn't fully answer the question, use your general knowledge but mention this
- Provide code examples when relevant, using proper markdown code blocks
- Be concise but thorough
- If you're unsure, say so
- When citing sources, reference them as [Source N]"""

    def _get_user_prompt(self, query: str, context: str) -> str:
        """Build the user prompt with context and query."""
        if context:
            return f"""Context from documentation:
{context}

Question: {query}"""
        else:
            return query

    def _build_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        """Build messages list with proper system/user role separation."""
        return [
            {'role': 'system', 'content': self._get_system_prompt()},
            {'role': 'user', 'content': self._get_user_prompt(query, context)}
        ]
    
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
        
        # Build messages with proper system/user separation
        messages = self._build_messages(query, context_str)

        # Generate response using Ollama
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
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

    def _run_ollama_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Run synchronous Ollama streaming in a thread and put results in async queue.

        This method runs in a separate thread to avoid blocking the event loop.
        """
        try:
            stream = ollama.chat(
                model=model,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                },
                stream=True
            )

            for chunk in stream:
                if chunk.get('message', {}).get('content'):
                    # Thread-safe way to put item in async queue
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {'type': 'content', 'content': chunk['message']['content']}
                    )

            # Signal completion
            loop.call_soon_threadsafe(queue.put_nowait, {'type': 'done'})

        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {'type': 'error', 'error': str(e)}
            )

    async def generate_response_stream(
        self,
        query: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_rag: bool = True,
    ):
        """Generate streaming response using RAG pipeline.

        This async generator properly yields control back to the event loop
        by running the synchronous Ollama streaming in a thread pool.
        """

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

        # Build messages with proper system/user separation
        messages = self._build_messages(query, context_str)

        # Format sources
        sources = []
        if context_docs:
            for doc in context_docs:
                sources.append({
                    'source': doc.metadata.get('source', 'Unknown'),
                    'source_type': doc.metadata.get('source_type', 'Unknown'),
                    'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                })

        # Yield metadata first
        yield {
            'type': 'metadata',
            'model': model,
            'context_used': bool(context_docs),
            'sources': sources if sources else None,
        }

        # Generate streaming response using Ollama in a thread pool
        # This prevents blocking the event loop
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Run the synchronous Ollama streaming in a thread pool
        thread_task = loop.run_in_executor(
            None,  # Use default executor (ThreadPoolExecutor)
            self._run_ollama_stream,
            model,
            messages,
            temperature,
            max_tokens,
            queue,
            loop,
        )

        # Consume chunks from the queue as they arrive
        try:
            while True:
                # Wait for next chunk from the thread, yielding control to event loop
                chunk = await queue.get()

                if chunk['type'] == 'done':
                    yield chunk
                    break
                elif chunk['type'] == 'error':
                    yield chunk
                    break
                else:
                    yield chunk
        finally:
            # Ensure the thread task completes
            await thread_task


# Singleton instance
rag_pipeline = RAGPipeline()
