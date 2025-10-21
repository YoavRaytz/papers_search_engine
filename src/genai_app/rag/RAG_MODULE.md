# RAG Module Documentation

## Overview

The `rag` module implements **Retrieval-Augmented Generation (RAG)** - combining semantic search with local LLM inference to generate contextually grounded answers from retrieved documents.

---

## Module Structure

```
rag/
├── __init__.py
└── llm_query.py        # RAG query system with LLM
```

---

## What is RAG?

**Retrieval-Augmented Generation** enhances LLM responses by:
1. **Retrieving** relevant documents from knowledge base
2. **Augmenting** prompt with retrieved context
3. **Generating** answers grounded in provided documents

```
User Query: "What is attention mechanism?"
    ↓
┌─────────────────────────┐
│  1. RETRIEVAL           │  → Search index for top 3 papers
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  2. AUGMENTATION        │  → Build prompt with:
│                         │     - System instructions
│                         │     - Retrieved papers
│                         │     - User query
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  3. GENERATION          │  → LLM generates answer
└─────────────────────────┘
    ↓
Answer: "The attention mechanism, introduced in 
'Attention Is All You Need', allows the model to..."
```

---

## Class: `RAGQuerySystem`

### Initialization

```python
class RAGQuerySystem:
    """Query answering using retrieved context + local LLM."""
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize LLM for text generation.
        Uses CPU if no GPU available.
        """
        device = 0 if torch.cuda.is_available() else -1
        
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float32 if device == -1 else torch.float16,
            device=device,
            max_new_tokens=512,
        )
```

**Model:** TinyLlama-1.1B (lightweight, fast CPU inference)

---

## Core Methods

### 1. `format_prompt()` - Prompt Engineering

Structures context and query for the LLM using chat format:

```python
def format_prompt(self, query: str, context_docs: List[Tuple[dict, float]]) -> str:
    """Format prompt with retrieved documents."""
    
    # Build context from documents
    context_parts = []
    for i, (doc, distance) in enumerate(context_docs, 1):
        context_parts.append(
            f"[Document {i} - ID: {doc['id']}]\n"
            f"Title: {doc['title']}\n"
            f"Abstract: {doc['abstract']}\n"
        )
    
    context_text = "\n".join(context_parts)
    
    # Chat format with special tokens
    prompt = f"""<|system|>
You are a helpful AI assistant that answers questions based on provided research papers. 
Use only the information from the given documents to answer the question.
</s>
<|user|>
Context from research papers:

{context_text}

Question: {query}
</s>
<|assistant|>
"""
    return prompt
```

**Example formatted prompt:**
```
<|system|>
You are a helpful AI assistant...
</s>
<|user|>
Context from research papers:

[Document 1 - ID: arxiv_1706.03762]
Title: Attention Is All You Need
Abstract: The dominant sequence transduction models...

[Document 2 - ID: arxiv_1810.04805]
Title: BERT: Pre-training of Deep Bidirectional Transformers
Abstract: We introduce a new language representation model...

Question: What is the attention mechanism in transformers?
</s>
<|assistant|>
```

---

### 2. `generate_answer()` - Standard Generation

Generates complete answer (blocking):

```python
def generate_answer(self, prompt: str) -> str:
    """Generate answer using LLM."""
    outputs = self.pipe(
        prompt,
        do_sample=True,
        temperature=0.7,        # Randomness control
        top_p=0.95,             # Nucleus sampling
        repetition_penalty=1.15, # Avoid repetition
    )
    
    full_output = outputs[0]['generated_text']
    
    # Extract assistant's response
    if '<|assistant|>' in full_output:
        answer = full_output.split('<|assistant|>')[-1].strip()
    else:
        answer = full_output[len(prompt):].strip()
    
    return answer.replace('</s>', '').strip()
```

---

### 3. `generate_answer_stream()` - Streaming Generation

Generates answer token-by-token (real-time):

```python
def generate_answer_stream(self, prompt: str) -> Iterator[str]:
    """Stream answer generation token by token."""
    
    # Setup streaming
    streamer = TextIteratorStreamer(
        self.pipe.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    # Generate in background thread
    generation_kwargs = {
        "text_inputs": prompt,
        "streamer": streamer,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.15,
        "max_new_tokens": 512,
    }
    
    thread = Thread(target=self.pipe.model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield tokens as they're generated
    for token in streamer:
        yield token
```

**Usage:**
```python
for token in rag.generate_answer_stream(prompt):
    print(token, end='', flush=True)
```

---

### 4. `answer_query()` - Complete RAG Pipeline

Combines everything into one method:

```python
def answer_query(self, query: str, context_docs: List[Tuple[dict, float]], 
                 stream: bool = False):
    """
    Complete RAG pipeline: format prompt → generate answer.
    
    Args:
        query: User question
        context_docs: Retrieved (document, distance) tuples
        stream: If True, return generator; if False, return dict
    """
    # Format prompt
    prompt = self.format_prompt(query, context_docs)
    
    # Prepare metadata
    sources = [{'id': doc['id'], 'title': doc['title'], 'distance': dist}
               for doc, dist in context_docs]
    
    citations = [{'doc_id': doc['id'], 'title': doc['title'], 
                  'citation': doc['abstract'][:200] + '...'}
                 for doc, dist in context_docs]
    
    # Generate
    if stream:
        return self._stream_with_metadata(prompt, query, sources, citations)
    else:
        answer = self.generate_answer(prompt)
        return {
            'query': query,
            'answer': answer,
            'sources': sources,
            'citations': citations,
            'num_sources': len(sources)
        }
```

**Non-streaming example:**
```python
result = rag.answer_query(query, context_docs, stream=False)

# Returns:
{
    'query': 'What is attention mechanism?',
    'answer': 'The attention mechanism, introduced in...',
    'sources': [
        {'id': 'arxiv_1706', 'title': 'Attention Is All You Need', 'distance': 2.34},
        {'id': 'arxiv_1810', 'title': 'BERT', 'distance': 3.12}
    ],
    'citations': [...],
    'num_sources': 2
}
```

**Streaming example:**
```python
stream = rag.answer_query(query, context_docs, stream=True)

for chunk in stream:
    if chunk['type'] == 'metadata':
        print(f"Sources: {chunk['num_sources']}")
    elif chunk['type'] == 'token':
        print(chunk['content'], end='', flush=True)
    elif chunk['type'] == 'done':
        print("\n[Complete]")
```

---

## Complete Workflow Example

```python
from genai_app.index.index_manager import IndexManager
from genai_app.rag.llm_query import RAGQuerySystem

# 1. Initialize
index_manager = IndexManager('data/arxiv_2.9k.jsonl')
index_manager.initialize()

rag_system = RAGQuerySystem()

# 2. User query
query = "How do transformers improve upon RNNs?"

# 3. Retrieve documents
context_docs = index_manager.search_with_documents(query, k=3)

# 4. Generate answer
result = rag_system.answer_query(query, context_docs, stream=False)

# 5. Display
print(f"Question: {result['query']}")
print(f"Answer: {result['answer']}")
print(f"\nSources ({result['num_sources']}):")
for src in result['sources']:
    print(f"  • {src['title']} (relevance: {1/(1+src['distance']):.0%})")
```

---

## Streaming in Web App

The Flask app uses Server-Sent Events (SSE) for streaming:

```python
# app.py
@app.route('/api/query/stream')
def query_stream():
    query = request.args.get('query')
    
    # Retrieve documents
    docs = index_manager.search_with_documents(query, k=3)
    
    # Stream answer
    def generate():
        stream = rag_system.answer_query(query, docs, stream=True)
        for chunk in stream:
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')
```

**Frontend (JavaScript):**
```javascript
const eventSource = new EventSource(`/api/query/stream?query=${query}`);

eventSource.onmessage = (event) => {
    const chunk = JSON.parse(event.data);
    
