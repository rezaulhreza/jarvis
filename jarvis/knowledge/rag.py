"""
RAG (Retrieval Augmented Generation) module for Jarvis.

Handles document ingestion, embedding, storage, and retrieval.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings


@dataclass
class Document:
    """A document chunk with metadata."""
    content: str
    source: str
    chunk_index: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RAGEngine:
    """RAG engine using ChromaDB for vector storage."""

    def __init__(self, persist_dir: str, embedding_model: str = "nomic-embed-text", ollama_client=None):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model
        self.ollama_client = ollama_client

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Default collection for documents
        # Use cosine distance for better relevance scoring (0-2 range, lower = more similar)
        self.collection = self.client.get_or_create_collection(
            name="jarvis_knowledge",
            metadata={"hnsw:space": "cosine", "description": "Jarvis knowledge base"}
        )

    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding using Ollama."""
        if self.ollama_client:
            return self.ollama_client.embed(text, model=self.embedding_model)

        # Fallback to direct Ollama API call
        import requests
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": self.embedding_model, "prompt": text}
        )
        return response.json()["embedding"]

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        if len(words) <= chunk_size:
            return [text]

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap

        return chunks

    def _generate_id(self, source: str, chunk_index: int) -> str:
        """Generate unique ID for a document chunk."""
        content = f"{source}:{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()

    def add_document(self, content: str, source: str, metadata: dict = None) -> int:
        """Add a document to the knowledge base.

        Args:
            content: Document text content
            source: Source identifier (filename, URL, etc.)
            metadata: Optional metadata dict

        Returns:
            Number of chunks added
        """
        chunks = self._chunk_text(content)

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = self._generate_id(source, i)
            embedding = self._get_embedding(chunk)

            meta = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {})
            }

            ids.append(doc_id)
            embeddings.append(embedding)
            documents.append(chunk)
            metadatas.append(meta)

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return len(chunks)

    def add_file(self, file_path: str, metadata: dict = None) -> int:
        """Add a file to the knowledge base.

        Supports: .txt, .md, .pdf
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()

        if ext in [".txt", ".md"]:
            content = path.read_text(encoding="utf-8")
        elif ext == ".pdf":
            content = self._extract_pdf(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        source = path.name
        return self.add_document(content, source, metadata)

    def _extract_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return "\n".join(text)
        except ImportError:
            raise ImportError("pypdf is required for PDF support. Install with: pip install pypdf")

    def add_directory(self, dir_path: str, extensions: list[str] = None) -> dict:
        """Add all documents from a directory.

        Args:
            dir_path: Directory path
            extensions: File extensions to include (default: ['.txt', '.md', '.pdf'])

        Returns:
            Dict with results per file
        """
        if extensions is None:
            extensions = [".txt", ".md", ".pdf"]

        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        results = {}
        for ext in extensions:
            for file_path in path.rglob(f"*{ext}"):
                try:
                    chunks = self.add_file(str(file_path))
                    results[str(file_path)] = {"status": "success", "chunks": chunks}
                except Exception as e:
                    results[str(file_path)] = {"status": "error", "error": str(e)}

        return results

    def search(self, query: str, n_results: int = 5, filter_metadata: dict = None) -> list[dict]:
        """Search for relevant documents.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of results with content, source, and score
        """
        query_embedding = self._get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )

        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                documents.append({
                    "content": doc,
                    "source": results["metadatas"][0][i].get("source", "unknown"),
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None
                })

        return documents

    def get_context(self, query: str, n_results: int = 3, max_tokens: int = 1500) -> str:
        """Get formatted context for injection into prompts.

        Args:
            query: User query
            n_results: Number of documents to retrieve
            max_tokens: Approximate max tokens for context

        Returns:
            Formatted context string
        """
        results = self.search(query, n_results=n_results)

        if not results:
            return ""

        context_parts = []
        total_len = 0

        for doc in results:
            # Rough token estimate (4 chars per token)
            doc_len = len(doc["content"]) // 4
            if total_len + doc_len > max_tokens:
                break

            source = doc["source"]
            content = doc["content"].strip()
            context_parts.append(f"[From: {source}]\n{content}")
            total_len += doc_len

        if not context_parts:
            return ""

        return "Relevant knowledge:\n\n" + "\n\n---\n\n".join(context_parts)

    def delete_source(self, source: str) -> int:
        """Delete all chunks from a specific source.

        Args:
            source: Source identifier to delete

        Returns:
            Number of chunks deleted
        """
        # Get all IDs for this source
        results = self.collection.get(
            where={"source": source}
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])

        return 0

    def list_sources(self) -> list[dict]:
        """List all document sources in the knowledge base.

        Returns:
            List of sources with chunk counts
        """
        # Get all documents
        results = self.collection.get()

        sources = {}
        for meta in results["metadatas"]:
            source = meta.get("source", "unknown")
            if source not in sources:
                sources[source] = {"source": source, "chunks": 0}
            sources[source]["chunks"] += 1

        return list(sources.values())

    def count(self) -> int:
        """Get total number of document chunks."""
        return self.collection.count()

    def clear(self) -> int:
        """Clear all documents from the knowledge base.

        Returns:
            Number of chunks deleted
        """
        count = self.collection.count()
        if count > 0:
            # Delete the collection and recreate it
            self.client.delete_collection("jarvis_knowledge")
            self.collection = self.client.create_collection(
                name="jarvis_knowledge",
                metadata={"description": "Jarvis knowledge base"}
            )
        return count


# Convenience function for getting RAG engine
_rag_engine: Optional[RAGEngine] = None

def get_rag_engine(config: dict = None) -> RAGEngine:
    """Get or create the RAG engine singleton."""
    global _rag_engine

    if _rag_engine is None:
        if config is None:
            from jarvis.assistant import load_config
            config = load_config()

        persist_dir = config.get("memory", {}).get("vector_store", "knowledge/chroma_db")
        embedding_model = config.get("models", {}).get("embeddings", "nomic-embed-text")

        # Make path absolute if relative
        if not os.path.isabs(persist_dir):
            base_dir = Path(__file__).parent.parent.parent
            persist_dir = str(base_dir / persist_dir)

        _rag_engine = RAGEngine(persist_dir, embedding_model)

    return _rag_engine
