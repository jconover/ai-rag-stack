#!/usr/bin/env python3
"""
DevOps Documentation Ingestion Pipeline
Processes markdown/text files and indexes them in Qdrant vector database
"""

import os
import sys
from pathlib import Path
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from tqdm import tqdm

# Configuration
DOCS_DIR = os.getenv("DOCS_DIR", "../data/docs")
CUSTOM_DOCS_DIR = os.getenv("CUSTOM_DOCS_DIR", "../data/custom")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "devops_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))


class DocumentIngestionPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Change to 'cuda' if you want GPU embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
    def load_documents_from_directory(self, directory: str, source_name: str) -> List:
        """Load all markdown and text files from a directory"""
        documents = []
        
        # Load markdown files
        md_loader = DirectoryLoader(
            directory,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True,
            use_multithreading=True,
            max_concurrency=MAX_WORKERS,
        )
        
        try:
            md_docs = md_loader.load()
            for doc in md_docs:
                doc.metadata['source_type'] = source_name
            documents.extend(md_docs)
            print(f"Loaded {len(md_docs)} markdown files from {source_name}")
        except Exception as e:
            print(f"Error loading markdown from {directory}: {e}")
        
        # Load text files
        txt_loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
        )
        
        try:
            txt_docs = txt_loader.load()
            for doc in txt_docs:
                doc.metadata['source_type'] = source_name
            documents.extend(txt_docs)
            print(f"Loaded {len(txt_docs)} text files from {source_name}")
        except Exception as e:
            print(f"Error loading text from {directory}: {e}")
        
        return documents
    
    def ingest_documents(self, documents: List, collection_name: str = COLLECTION_NAME):
        """Split documents and ingest into Qdrant"""
        if not documents:
            print("No documents to ingest")
            return
        
        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        print(f"Ingesting into Qdrant collection '{collection_name}'...")
        
        # Create or update vector store
        vectorstore = Qdrant.from_documents(
            chunks,
            self.embeddings,
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=collection_name,
            force_recreate=False,  # Set to True to recreate collection
        )
        
        print(f"Successfully ingested {len(chunks)} chunks into Qdrant")
        return vectorstore
    
    def run(self):
        """Main ingestion pipeline"""
        all_documents = []
        
        # Documentation sources to ingest
        doc_sources = {
            "kubernetes": os.path.join(DOCS_DIR, "kubernetes"),
            "kubernetes-ai": os.path.join(DOCS_DIR, "kubernetes-ai"),
            "terraform": os.path.join(DOCS_DIR, "terraform"),
            "docker": os.path.join(DOCS_DIR, "docker"),
            "ansible": os.path.join(DOCS_DIR, "ansible"),
            "prometheus": os.path.join(DOCS_DIR, "prometheus"),
            "custom": CUSTOM_DOCS_DIR,
        }
        
        # Load documents from each source
        for source_name, directory in doc_sources.items():
            if os.path.exists(directory):
                print(f"\n{'='*60}")
                print(f"Processing {source_name} documentation...")
                print(f"{'='*60}")
                docs = self.load_documents_from_directory(directory, source_name)
                all_documents.extend(docs)
            else:
                print(f"Directory not found: {directory} (skipping {source_name})")
        
        # Ingest all documents
        if all_documents:
            print(f"\n{'='*60}")
            print(f"Total documents loaded: {len(all_documents)}")
            print(f"{'='*60}\n")
            self.ingest_documents(all_documents)
        else:
            print("No documents found to ingest!")
            sys.exit(1)


def main():
    print("Starting DevOps Documentation Ingestion Pipeline...")
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}\n")
    
    pipeline = DocumentIngestionPipeline()
    pipeline.run()
    
    print("\nIngestion complete!")


if __name__ == "__main__":
    main()
