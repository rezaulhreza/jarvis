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
              filter_metadata: dict = None, query_text: str = None) -> dict:
        """Query for similar documents. query_text enables hybrid search if supported."""
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
              filter_metadata: dict = None, query_text: str = None) -> dict:
        # query_text ignored - ChromaDB doesn't support hybrid search
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


class SparseEncoder:
    """Simple BM25-style sparse encoder for keyword matching."""

    def __init__(self):
        self.vocab: dict[str, int] = {}  # word -> index
        self.idf: dict[str, float] = {}  # word -> IDF score
        self.doc_count = 0

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        # Remove very short tokens and stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'to', 'of', 'and', 'in', 'that', 'it', 'for', 'on', 'with',
                     'as', 'at', 'by', 'from', 'or', 'this', 'which', 'you', 'we'}
        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    def fit(self, documents: list[str]):
        """Build vocabulary and IDF from documents."""
        import math
        self.doc_count = len(documents)
        doc_freq: dict[str, int] = {}

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Calculate IDF
        for token, df in doc_freq.items():
            self.idf[token] = math.log((self.doc_count + 1) / (df + 1)) + 1

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        """Encode text to sparse vector (indices, values)."""
        from collections import Counter
        tokens = self._tokenize(text)
        tf = Counter(tokens)

        indices = []
        values = []

        for token, count in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                # TF-IDF score
                tf_score = 1 + (count / len(tokens)) if tokens else 0
                idf_score = self.idf.get(token, 1.0)
                score = tf_score * idf_score

                indices.append(idx)
                values.append(score)

        # Sort by index for Qdrant
        if indices:
            sorted_pairs = sorted(zip(indices, values))
            indices, values = zip(*sorted_pairs)
            return list(indices), list(values)

        return [], []


class QdrantVectorStore(VectorStore):
    """Qdrant vector store backend with hybrid search (dense + sparse)."""

    def __init__(self, url: str, api_key: str, collection_name: str = "jarvis_knowledge",
                 hybrid: bool = True):
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "qdrant-client is required for Qdrant backend. "
                "Install with: pip install qdrant-client"
            )

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.hybrid = hybrid
        self.sparse_encoder = SparseEncoder() if hybrid else None
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        from qdrant_client.models import (
            Distance, VectorParams, PayloadSchemaType,
            SparseVectorParams, SparseIndexParams
        )

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            if self.hybrid:
                # Hybrid collection with both dense and sparse vectors
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=768, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        )
                    }
                )
            else:
                # Dense-only collection
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
        from qdrant_client.models import PointStruct, SparseVector

        # Fit sparse encoder on documents if hybrid
        if self.hybrid and self.sparse_encoder:
            self.sparse_encoder.fit(documents)

        points = []
        for doc_id, embedding, document, metadata in zip(
            ids, embeddings, documents, metadatas
        ):
            numeric_id = int(hashlib.md5(doc_id.encode()).hexdigest()[:16], 16)

            if self.hybrid and self.sparse_encoder:
                # Hybrid: both dense and sparse vectors
                sparse_indices, sparse_values = self.sparse_encoder.encode(document)
                point = PointStruct(
                    id=numeric_id,
                    vector={
                        "dense": embedding,
                        "sparse": SparseVector(indices=sparse_indices, values=sparse_values)
                    },
                    payload={
                        "document": document,
                        "doc_id": doc_id,
                        **metadata
                    }
                )
            else:
                # Dense only
                point = PointStruct(
                    id=numeric_id,
                    vector=embedding,
                    payload={
                        "document": document,
                        "doc_id": doc_id,
                        **metadata
                    }
                )
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, embedding: list[float], n_results: int = 5,
              filter_metadata: dict = None, query_text: str = None) -> dict:
        """Query with hybrid search (dense + sparse with RRF fusion)."""
        from qdrant_client.models import (
            Filter, FieldCondition, MatchValue,
            Prefetch, FusionQuery, Fusion, SparseVector
        )

        # Build filter if provided
        qdrant_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_metadata.items()
            ]
            qdrant_filter = Filter(must=conditions)

        if self.hybrid and query_text and self.sparse_encoder:
            # Hybrid search with RRF fusion
            sparse_indices, sparse_values = self.sparse_encoder.encode(query_text)

            if sparse_indices:
                # Use prefetch for both dense and sparse, then fuse with RRF
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        Prefetch(
                            query=embedding,
                            using="dense",
                            limit=n_results * 2,
                            filter=qdrant_filter
                        ),
                        Prefetch(
                            query=SparseVector(indices=sparse_indices, values=sparse_values),
                            using="sparse",
                            limit=n_results * 2,
                            filter=qdrant_filter
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=n_results,
                    with_payload=True
                )
            else:
                # No sparse matches, fall back to dense only
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=embedding,
                    using="dense",
                    limit=n_results,
                    query_filter=qdrant_filter,
                    with_payload=True
                )
        else:
            # Dense-only search
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                using="dense" if self.hybrid else None,
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
            "distances": [1 - r.score if r.score else 0 for r in results]
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


class Reranker:
    """Cross-encoder reranker for improving retrieval quality."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for reranking. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """Rerank documents by relevance to query using cross-encoder.

        Args:
            query: The search query
            documents: List of document dicts with 'content' key
            top_k: Number of top documents to return

        Returns:
            Reranked documents (top_k most relevant)
        """
        if not documents:
            return []

        model = self._load_model()

        # Create query-document pairs for scoring
        pairs = [(query, doc["content"]) for doc in documents]

        # Get relevance scores from cross-encoder
        scores = model.predict(pairs)

        # Attach scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort by rerank score (higher = more relevant)
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]


class RAGEngine:
    """RAG engine with pluggable vector store backends.

    Features:
    - Pluggable vector stores (ChromaDB, Qdrant)
    - Optional reranking with cross-encoder models
    - Configurable chunking and retrieval
    """

    def __init__(self, vector_store: VectorStore, embedding_model: str = "nomic-embed-text",
                 ollama_client=None, reranker: Reranker = None):
        self.store = vector_store
        self.embedding_model = embedding_model
        self.ollama_client = ollama_client
        self.reranker = reranker

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

    def search(self, query: str, n_results: int = 5, filter_metadata: dict = None,
                rerank: bool = True) -> list[dict]:
        """Search for relevant documents with optional reranking.

        Two-stage retrieval:
        1. Fast vector search retrieves candidates (4x requested if reranking)
        2. Cross-encoder reranks candidates by relevance (if reranker enabled)

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            rerank: Whether to use reranking (default True if reranker available)

        Returns:
            List of relevant documents sorted by relevance
        """
        query_embedding = self._get_embedding(query)

        # If reranking, retrieve more candidates for better recall
        retrieve_count = n_results * 4 if (self.reranker and rerank) else n_results

        results = self.store.query(
            embedding=query_embedding,
            n_results=retrieve_count,
            filter_metadata=filter_metadata,
            query_text=query  # For hybrid search (keyword matching)
        )

        documents = []
        for i, doc in enumerate(results["documents"]):
            documents.append({
                "content": doc,
                "source": results["metadatas"][i].get("source", "unknown"),
                "metadata": results["metadatas"][i],
                "distance": results["distances"][i] if results["distances"] else None
            })

        # Apply reranking if enabled
        if self.reranker and rerank and documents:
            documents = self.reranker.rerank(query, documents, top_k=n_results)

        return documents[:n_results]

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


def create_reranker(config: dict) -> Optional[Reranker]:
    """Create a reranker if enabled in config or environment.

    Enable reranking by:
    - Setting RAG_RERANK=true in environment, or
    - Setting memory.rerank: true in settings.yaml
    """
    # Check environment first
    env_rerank = os.environ.get("RAG_RERANK", "").lower()
    if env_rerank in ("true", "1", "yes"):
        return Reranker()

    # Check config
    if config.get("memory", {}).get("rerank", False):
        rerank_model = config.get("memory", {}).get(
            "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        return Reranker(model_name=rerank_model)

    return None


def get_rag_engine(config: dict = None) -> RAGEngine:
    """Get or create the RAG engine singleton."""
    global _rag_engine

    if _rag_engine is None:
        if config is None:
            from jarvis.assistant import load_config
            config = load_config()

        embedding_model = config.get("models", {}).get("embeddings", "nomic-embed-text")
        vector_store = create_vector_store(config)
        reranker = create_reranker(config)

        _rag_engine = RAGEngine(vector_store, embedding_model, reranker=reranker)

    return _rag_engine
