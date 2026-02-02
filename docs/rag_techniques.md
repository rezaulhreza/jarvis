# RAG Optimization Techniques

This document covers advanced techniques for improving Retrieval-Augmented Generation (RAG) quality in Jarvis.

## Overview

Basic RAG has a simple flow:
```
Query → Embed → Vector Search → Top-K Results → LLM
```

Advanced RAG adds optimization stages:
```
Query → [Query Expansion] → Embed → Vector Search → [Reranking] → [Compression] → LLM
```

---

## Implemented Techniques

### 1. Two-Stage Retrieval with Reranking

**Problem:** Vector similarity search is fast but approximate. Embeddings capture semantic meaning but may miss nuanced relevance.

**Solution:** Retrieve more candidates, then rerank with a more accurate (but slower) model.

```
Stage 1: Vector Search (fast, approximate)
    Query → Embedding → Top 20 candidates

Stage 2: Cross-Encoder Reranking (slow, accurate)
    Query + Each Candidate → Relevance Score → Top 5
```

**How it works in Jarvis:**

```python
# jarvis/knowledge/rag.py

class Reranker:
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list, top_k: int = 5):
        # Score each query-document pair
        pairs = [(query, doc["content"]) for doc in documents]
        scores = self.model.predict(pairs)

        # Sort by relevance score
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:top_k]]
```

**Configuration:**

```yaml
# config/settings.yaml
memory:
  rerank: true
  rerank_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

Or via environment:
```bash
export RAG_RERANK=true
```

**Why cross-encoders are better for reranking:**

| Bi-Encoder (Embeddings) | Cross-Encoder (Reranking) |
|------------------------|---------------------------|
| Encodes query and doc separately | Encodes query+doc together |
| Fast: O(1) per comparison | Slow: O(n) per comparison |
| Good for initial retrieval | Better for final ranking |
| Captures general similarity | Captures specific relevance |

**Impact:** Typically improves accuracy by 10-20% on retrieval benchmarks.

---

## Implemented Techniques (continued)

### 2. Hybrid Search (BM25 + Vector) - IMPLEMENTED

**Problem:** Pure vector search misses exact keyword matches. Pure keyword search misses semantic similarity.

**Solution:** Combine sparse (BM25/keyword) and dense (vector) retrieval with Reciprocal Rank Fusion.

```
┌─────────────────────────────────────────────┐
│                  Query                       │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│  BM25 Search  │   │ Vector Search │
│  (keywords)   │   │  (semantic)   │
└───────┬───────┘   └───────┬───────┘
        │                   │
        └─────────┬─────────┘
                  ▼
        ┌─────────────────┐
        │ Reciprocal Rank │
        │     Fusion      │
        └────────┬────────┘
                 ▼
           Final Results
```

**Reciprocal Rank Fusion (RRF):**
```python
def rrf_score(rank, k=60):
    return 1 / (k + rank)

# Combine rankings
final_score = rrf_score(bm25_rank) + rrf_score(vector_rank)
```

**Why it helps:**
- BM25 finds "Laravel developer" when query is "Laravel developer"
- Vectors find "PHP framework expert" for same query
- Combined: "What is my website" matches both semantically AND the keyword "website"

**Implementation in Jarvis:**

```python
# jarvis/knowledge/rag.py

class SparseEncoder:
    """Simple BM25-style sparse encoder for keyword matching."""

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        """Encode text to sparse vector (indices, values)."""
        tokens = self._tokenize(text)
        tf = Counter(tokens)

        indices, values = [], []
        for token, count in tf.items():
            if token in self.vocab:
                tf_score = 1 + (count / len(tokens))
                idf_score = self.idf.get(token, 1.0)
                indices.append(self.vocab[token])
                values.append(tf_score * idf_score)

        return indices, values


class QdrantVectorStore:
    def query(self, embedding, query_text, n_results=5):
        # Hybrid search with RRF fusion
        sparse_indices, sparse_values = self.sparse_encoder.encode(query_text)

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=embedding, using="dense", limit=n_results * 2),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    limit=n_results * 2
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
            limit=n_results,
        )
```

**Enabled by default** when using Qdrant backend. No configuration needed
- Combined: best of both worlds

---

### 3. Relevance Filtering via Rerank Scores - IMPLEMENTED

**Problem:** RAG retrieves chunks for every query, even when irrelevant. Heuristic-based query classification (pattern matching) is brittle and misclassifies many queries.

**Solution:** Always retrieve, then filter based on cross-encoder rerank scores:

```
┌─────────────────────────────────────────────┐
│                  Query                       │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │  Vector Search  │  Retrieve candidates
        │  + Hybrid BM25  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Cross-Encoder   │  Score relevance
        │   Reranking     │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Score Threshold │  Filter by rerank_score >= 0.0
        └────────┬────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Results ≥ 0.0      Results < 0.0
        │                 │
        ▼                 ▼
   Include in         Discard
   context            (use LLM knowledge)
```

**Why rerank scores are better than query classification:**
- Cross-encoder scores (-10 to +10) directly measure query-document relevance
- No brittle pattern matching ("explain X" no longer skips personal queries)
- Works for any query type - the relevance score decides, not heuristics
- Configurable threshold (`rerank_threshold: 0.0` in settings.yaml)

**Configuration:**
```yaml
# config/settings.yaml
memory:
  rerank: true
  rerank_threshold: 0.0  # Min score to include results (-10 to +10)
```

**Debug logging:**
```bash
export RAG_DEBUG=true  # See rerank scores in logs
```

**Impact:** All queries go through retrieval. Only genuinely relevant results are included. No more missed personal queries due to pattern matching.

---

### 4. Query Expansion

**Problem:** User queries are often short or ambiguous. "auth issues" could mean authentication, authorization, or authentication errors.

**Solution:** Generate multiple query variations before searching.

```python
def expand_query(query: str, llm) -> list[str]:
    """Generate query variations using LLM."""
    prompt = f"""Generate 3 search variations for: "{query}"
    Return only the variations, one per line."""

    variations = llm.generate(prompt).split('\n')
    return [query] + variations  # Include original

# Example:
# Input: "auth issues"
# Output: ["auth issues", "authentication problems", "login errors", "authorization failures"]
```

**Search with expansion:**
```python
def search_with_expansion(query, n_results=5):
    queries = expand_query(query)
    all_results = []

    for q in queries:
        results = vector_search(q, n_results=n_results)
        all_results.extend(results)

    # Deduplicate and rank
    return dedupe_and_rank(all_results)
```

---

### 5. Semantic Chunking

**Problem:** Fixed-size chunks split documents arbitrarily, potentially cutting sentences or concepts in half.

**Solution:** Split at natural semantic boundaries (sentences, paragraphs, sections).

**Current (fixed-size):**
```python
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    # Blindly splits every 500 words
```

**Improved (semantic):**
```python
def semantic_chunk(text, max_chunk_size=500):
    """Split at sentence boundaries, respecting max size."""
    import nltk
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence.split())

        if current_size + sentence_size > max_chunk_size:
            # Start new chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
```

**Advanced: Hierarchical chunking for documents with structure:**
```python
def hierarchical_chunk(markdown_text):
    """Split by headers, preserving document structure."""
    sections = split_by_headers(markdown_text)

    for section in sections:
        yield {
            "content": section.content,
            "metadata": {
                "header": section.header,
                "level": section.level,
                "parent": section.parent_header
            }
        }
```

---

### 6. Contextual Compression

**Problem:** Retrieved chunks contain irrelevant information. A 500-word chunk might have only 2 relevant sentences.

**Solution:** Extract only relevant parts from each chunk before sending to LLM.

```python
def compress_context(query: str, documents: list, llm) -> str:
    """Extract relevant sentences from documents."""
    compressed = []

    for doc in documents:
        prompt = f"""Given the question: "{query}"

Extract ONLY the sentences from this document that help answer the question.
If nothing is relevant, return "NOT_RELEVANT".

Document:
{doc['content']}

Relevant sentences:"""

        result = llm.generate(prompt)
        if result != "NOT_RELEVANT":
            compressed.append(f"[From: {doc['source']}]\n{result}")

    return "\n\n".join(compressed)
```

**Benefits:**
- Reduces context length (saves tokens)
- Removes noise from LLM input
- Improves answer quality

---

## Technique Comparison

| Technique | Impact | Latency Cost | Implementation Effort | Status |
|-----------|--------|--------------|----------------------|--------|
| **Reranking** | High (10-20%) | Medium (+100-300ms) | Low | ✅ Implemented |
| **Hybrid Search** | High (15-25%) | Low (+50ms) | Medium | ✅ Implemented |
| **Rerank Score Filtering** | High (noise elimination) | None (uses rerank) | Low | ✅ Implemented |
| **Prompt Injection Hardening** | Security | None | Low | ✅ Implemented |
| **Query Expansion** | Medium (5-15%) | High (+LLM call) | Low | Not implemented |
| **Semantic Chunking** | Medium (5-10%) | None (indexing only) | Medium | Not implemented |
| **Contextual Compression** | Medium (10-15%) | High (+LLM calls) | Low | Not implemented |

---

## Recommended Implementation Order

1. **Reranking** (implemented) - Quick win, big impact
2. **Hybrid Search** (implemented) - Qdrant supports this natively with sparse encoder persistence
3. **Rerank Score Filtering** (implemented) - Threshold on cross-encoder scores, not brittle heuristics
4. **Semantic Chunking** - Improves index quality
5. **Query Expansion** - Good for ambiguous queries
6. **Contextual Compression** - Polish for production

---

## Monitoring RAG Quality

### Key Metrics

1. **Retrieval Recall@K**: Do the top K results contain the answer?
2. **Retrieval Precision@K**: What % of top K results are relevant?
3. **MRR (Mean Reciprocal Rank)**: How high is the first relevant result?
4. **Answer Quality**: Does the final answer correctly use retrieved context?

### Simple Evaluation Script

```python
def evaluate_rag(test_cases):
    """Evaluate RAG on test question-answer pairs."""
    results = []

    for query, expected_source in test_cases:
        retrieved = rag.search(query, n_results=5)
        sources = [doc['source'] for doc in retrieved]

        results.append({
            'query': query,
            'expected': expected_source,
            'found': expected_source in sources,
            'rank': sources.index(expected_source) + 1 if expected_source in sources else -1
        })

    recall = sum(r['found'] for r in results) / len(results)
    mrr = sum(1/r['rank'] for r in results if r['rank'] > 0) / len(results)

    print(f"Recall@5: {recall:.2%}")
    print(f"MRR: {mrr:.2f}")
```

---

## References

- [RAG Performance Optimization - DEV Community](https://dev.to/jamesli/rag-performance-optimization-engineering-practice-implementation-guide-based-on-langchain-34ej)
- [Advanced Retrieval Techniques - Medium](https://medium.com/@abhiragkulkarni12/advanced-retrieval-techniques-in-langchain-to-improve-the-efficiency-of-rag-systems-32b88d78383d)
- [Enhancing RAG with Reranking - MyScale](https://www.myscale.com/blog/maximizing-advanced-rag-models-langchain-reranking-techniques/)
- [Cross-Encoders - Sentence Transformers](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Qdrant Hybrid Search](https://qdrant.tech/documentation/concepts/hybrid-queries/)
