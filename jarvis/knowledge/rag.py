"""
RAG (Retrieval Augmented Generation) module for Jarvis.

Handles document ingestion, embedding, storage, and retrieval.
Supports multiple vector store backends: ChromaDB (default) and Qdrant (cloud).
"""

import os
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


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


class VectorStore(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    def add(self, ids: list[str], embeddings: list[list[float]],
            documents: list[str], metadatas: list[dict]) -> None:
        """Add documents to the store."""
        pass

    @abstractmethod
    def query(self, embedding: list[float], n_results: int = 5,
              filter_metadata: dict = None) -> dict:
        """Query for similar documents."""
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        pass

    @abstractmethod
    def get(self, where: dict = None) -> dict:
        """Get documents matching filter."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total document count."""
        pass

    @abstractmethod
    def clear(self) -> int:
        """Clear all documents. Returns count deleted."""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store backend (local, default)."""

    def __init__(self, persist_dir: str, collection_name: str = "jarvis_knowledge"):
        import chromadb
        from chromadb.config import Settings

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "description": "Jarvis knowledge base"}
        )
        self.collection_name = collection_name

    def add(self, ids: list[str], embeddings: list[list[float]],
            documents: list[str], metadatas: list[dict]) -> None:
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, embedding: list[float], n_results: int = 5,
              filter_metadata: dict = None) -> dict:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=filter_metadata
        )
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)

    def get(self, where: dict = None) -> dict:
        results = self.collection.get(where=where)
        return {
            "ids": results["ids"],
            "documents": results["documents"],
            "metadatas": results["metadatas"]
        }

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> int:
        count = self.collection.count()
        if count > 0:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "description": "Jarvis knowledge base"}
            )
        return count


class QdrantVectorStore(VectorStore):
    """Qdrant vector store backend (cloud)."""

    def __init__(self, url: str, api_key: str, collection_name: str = "jarvis_knowledge"):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client is required for Qdrant backend. "
                "Install with: pip install qdrant-client"
            )

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            # nomic-embed-text produces 768-dimensional vectors
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            # Create payload indexes for filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema=PayloadSchemaType.KEYWORD
            )

    def add(self, ids: list[str], embeddings: list[list[float]],
            documents: list[str], metadatas: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for doc_id, embedding, document, metadata in zip(
            ids, embeddings, documents, metadatas
        ):
            # Qdrant needs numeric IDs, so we hash the string ID
            numeric_id = int(hashlib.md5(doc_id.encode()).hexdigest()[:16], 16)
            points.append(PointStruct(
                id=numeric_id,
                vector=embedding,
                payload={
                    "document": document,
                    "doc_id": doc_id,
                    **metadata
                }
            ))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, embedding: list[float], n_results: int = 5,
              filter_metadata: dict = None) -> dict:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build filter if provided
        qdrant_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_metadata.items()
            ]
            qdrant_filter = Filter(must=conditions)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=n_results,
            query_filter=qdrant_filter,
            with_payload=True
        )

        results = response.points
        return {
            "ids": [r.payload.get("doc_id", str(r.id)) for r in results],
            "documents": [r.payload.get("document", "") for r in results],
            "metadatas": [{k: v for k, v in r.payload.items()
                         if k not in ("document", "doc_id")} for r in results],
            "distances": [1 - r.score for r in results]  # Convert similarity to distance
        }

    def delete(self, ids: list[str]) -> None:
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=ids))]
            )
        )

    def get(self, where: dict = None) -> dict:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if where:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in where.items()
            ]
            qdrant_filter = Filter(must=conditions)

        # Scroll through all matching points
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_filter,
            limit=10000,
            with_payload=True
        )

        return {
            "ids": [r.payload.get("doc_id", str(r.id)) for r in results],
            "documents": [r.payload.get("document", "") for r in results],
            "metadatas": [{k: v for k, v in r.payload.items()
                         if k not in ("document", "doc_id")} for r in results]
        }

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def clear(self) -> int:
        count = self.count()
        if count > 0:
            from qdrant_client.models import Distance, VectorParams
            self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        return count


class RAGEngine:
    """RAG engine with pluggable vector store backends."""

    def __init__(self, vector_store: VectorStore, embedding_model: str = "nomic-embed-text",
                 ollama_client=None):
        self.store = vector_store
        self.embedding_model = embedding_model
        self.ollama_client = ollama_client

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
        """Add a document to the knowledge base."""
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

        self.store.add(ids=ids, embeddings=embeddings,
                       documents=documents, metadatas=metadatas)

        return len(chunks)

    def add_file(self, file_path: str, metadata: dict = None) -> int:
        """Add a file to the knowledge base."""
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
        """Add all documents from a directory."""
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
        """Search for relevant documents."""
        query_embedding = self._get_embedding(query)

        results = self.store.query(
            embedding=query_embedding,
            n_results=n_results,
            filter_metadata=filter_metadata
        )

        documents = []
        for i, doc in enumerate(results["documents"]):
            documents.append({
                "content": doc,
                "source": results["metadatas"][i].get("source", "unknown"),
                "metadata": results["metadatas"][i],
                "distance": results["distances"][i] if results["distances"] else None
            })

        return documents

    def get_context(self, query: str, n_results: int = 5, max_tokens: int = 1500) -> str:
        """Get formatted context for injection into prompts."""
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
        """Delete all chunks from a specific source."""
        results = self.store.get(where={"source": source})

        if results["ids"]:
            self.store.delete(ids=results["ids"])
            return len(results["ids"])

        return 0

    def list_sources(self) -> list[dict]:
        """List all document sources in the knowledge base."""
        results = self.store.get()

        sources = {}
        for meta in results["metadatas"]:
            source = meta.get("source", "unknown")
            if source not in sources:
                sources[source] = {"source": source, "chunks": 0}
            sources[source]["chunks"] += 1

        return list(sources.values())

    def count(self) -> int:
        """Get total number of document chunks."""
        return self.store.count()

    def clear(self) -> int:
        """Clear all documents from the knowledge base."""
        return self.store.clear()


# Factory function and singleton
_rag_engine: Optional[RAGEngine] = None


def create_vector_store(config: dict) -> VectorStore:
    """Create the appropriate vector store based on environment.

    Priority:
    1. If QDRANT_URL + QDRANT_API_KEY are set in env → use Qdrant
    2. Otherwise → fall back to ChromaDB (local)
    """
    # Check if Qdrant credentials are available
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    if qdrant_url and qdrant_api_key:
        # Qdrant is configured - use it
        return QdrantVectorStore(url=qdrant_url, api_key=qdrant_api_key)

    # Fall back to ChromaDB (local)
    persist_dir = config.get("memory", {}).get("vector_store", "knowledge/chroma_db")

    # Make path absolute if relative
    if not os.path.isabs(persist_dir):
        base_dir = Path(__file__).parent.parent.parent
        persist_dir = str(base_dir / persist_dir)

    return ChromaVectorStore(persist_dir=persist_dir)


def get_rag_engine(config: dict = None) -> RAGEngine:
    """Get or create the RAG engine singleton."""
    global _rag_engine

    if _rag_engine is None:
        if config is None:
            from jarvis.assistant import load_config
            config = load_config()

        embedding_model = config.get("models", {}).get("embeddings", "nomic-embed-text")
        vector_store = create_vector_store(config)

        _rag_engine = RAGEngine(vector_store, embedding_model)

    return _rag_engine
