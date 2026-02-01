# AI & LLM Concepts for Jarvis

## Large Language Models (LLMs)

### What is an LLM?
A Large Language Model is a neural network trained on vast amounts of text data. It learns patterns in language and can generate human-like text, answer questions, write code, and more.

### How LLMs Work
1. **Tokenization**: Text is split into tokens (words or subwords)
2. **Embedding**: Tokens are converted to numerical vectors
3. **Attention**: Model weighs importance of different tokens
4. **Generation**: Model predicts next token based on context
5. **Sampling**: Token is selected (temperature controls randomness)

### Key Parameters
- **Temperature**: 0 = deterministic, 1+ = creative/random
- **Top-p (nucleus sampling)**: Limits token selection to top probability mass
- **Max tokens**: Limits response length
- **Context window**: Maximum input + output tokens

### Local Models (Ollama)
Ollama runs LLMs locally on your machine:
- **llama3.2**: Meta's efficient chat model
- **qwen3**: Alibaba's multilingual model
- **deepseek-r1**: Reasoning-focused model
- **qwen2.5-coder**: Optimized for code

## RAG (Retrieval Augmented Generation)

### What is RAG?
RAG enhances LLM responses by retrieving relevant information from a knowledge base before generating a response.

### Why RAG?
- LLMs have knowledge cutoff dates
- LLMs can hallucinate facts
- RAG grounds responses in your actual documents
- Enables domain-specific knowledge

### RAG Pipeline
```
Query → Embed Query → Vector Search → Retrieve Chunks → Augment Prompt → Generate
```

### Components
1. **Document Loader**: Reads PDF, MD, TXT files
2. **Chunker**: Splits documents into manageable pieces
3. **Embedder**: Converts text to vectors (nomic-embed-text)
4. **Vector Store**: ChromaDB stores and searches vectors
5. **Retriever**: Finds relevant chunks for a query
6. **Generator**: LLM that produces final answer

### Chunking Strategies
- **Fixed size**: Split every N words/tokens
- **Overlap**: Chunks share some content for context
- **Semantic**: Split at natural boundaries (paragraphs, sections)

## Embeddings

### What are Embeddings?
Embeddings are dense vector representations of text. Similar texts have similar vectors (close in vector space).

### Use Cases
- **Semantic search**: Find similar documents
- **Clustering**: Group related content
- **Classification**: Categorize text
- **RAG retrieval**: Match queries to relevant chunks

### Embedding Models
- **nomic-embed-text**: Good general-purpose embeddings
- **OpenAI ada-002**: High quality, requires API
- **sentence-transformers**: Various open models

### Similarity Metrics
- **Cosine similarity**: Angle between vectors (most common)
- **Euclidean distance**: Straight-line distance
- **Dot product**: Magnitude-sensitive similarity

## Agents & Tool Calling

### What is an Agent?
An agent is an LLM that can take actions by calling tools/functions. It decides what to do based on the user's request.

### Agent Loop
```
1. User request
2. LLM decides: respond or use tool?
3. If tool: execute and get result
4. Feed result back to LLM
5. Repeat until task complete
6. Return final response
```

### Tool/Function Calling
Tools are functions the LLM can invoke:
```python
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return search_results
```

The LLM outputs structured JSON to call tools:
```json
{
  "tool": "web_search",
  "arguments": {"query": "latest news"}
}
```

### ReAct Pattern
Reasoning + Acting:
1. **Thought**: LLM explains its reasoning
2. **Action**: LLM calls a tool
3. **Observation**: Tool returns result
4. **Repeat** until answer is found

## Prompting

### System Prompt
Instructions that define the AI's behavior, personality, and constraints. Set once at the start of conversation.

### User Prompt
The actual user message/question.

### Prompt Engineering
Techniques to get better responses:
- **Be specific**: Clear, detailed instructions
- **Provide examples**: Show desired format (few-shot)
- **Role assignment**: "You are an expert in..."
- **Chain of thought**: "Think step by step"
- **Output format**: "Respond in JSON format"

## Context & Memory

### Context Window
The maximum amount of text (in tokens) an LLM can process at once.
- GPT-4: 128K tokens
- Llama 3.2: 128K tokens
- Local models: Often 4K-32K

### Conversation Memory
Strategies for long conversations:
- **Full history**: Keep all messages (hits context limit)
- **Sliding window**: Keep last N messages
- **Summarization**: Compress old messages
- **RAG memory**: Store/retrieve past conversations

## Text-to-Speech (TTS)

### How TTS Works
1. Text is analyzed for pronunciation
2. Converted to phonemes
3. Neural network generates audio waveform
4. Output as audio file or stream

### TTS Options
- **Browser**: Built-in, instant, robotic
- **Edge TTS**: Microsoft neural voices, free, good quality
- **ElevenLabs**: Premium, very natural, streaming

## Speech-to-Text (STT)

### How STT Works
1. Audio is captured from microphone
2. Converted to spectrogram
3. Neural network transcribes to text
4. Post-processing for punctuation/formatting

### STT Options
- **Browser Web Speech API**: Real-time, instant, less accurate
- **Whisper**: OpenAI model, batch processing, very accurate

## Vector Databases

### What is a Vector Database?
A database optimized for storing and searching high-dimensional vectors (embeddings).

### ChromaDB
The vector store used in Jarvis:
- Persistent storage
- Fast similarity search
- Metadata filtering
- Python-native

### Operations
- **Add**: Store vectors with metadata
- **Query**: Find similar vectors
- **Delete**: Remove by ID or filter
- **Update**: Modify existing entries

## Fine-Tuning vs RAG

### Fine-Tuning
- Trains the model on your data
- Changes model weights
- Requires significant compute
- Best for: style, format, specialized tasks

### RAG
- Retrieves relevant context at runtime
- Model weights unchanged
- No training required
- Best for: factual knowledge, documents

### When to Use What
- **RAG**: Current info, documents, facts
- **Fine-tuning**: Behavior, style, format
- **Both**: Complex domain-specific applications

## Inference Optimization

### Quantization
Reducing model precision (32-bit → 4-bit) to save memory and speed up inference.

### Batching
Processing multiple requests together for efficiency.

### KV Cache
Storing computed key-value pairs to avoid recomputation during generation.

### Speculative Decoding
Using a smaller model to draft, larger model to verify.
