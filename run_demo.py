#!/usr/bin/env python3
"""
GenAI RAG Application - Complete Demo

Demonstrates:
- Memory-efficient dataset loading (streaming)
- Vector index building with batch processing
- Semantic search across all documents
- RAG query answering with local LLM
"""
import os
from pathlib import Path
from src.genai_app.datasets.jsonl_dataset import iter_jsonl
from src.genai_app.index.index_manager import IndexManager
from src.genai_app.rag.llm_query import RAGQuerySystem


def print_header(title):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")


def demo_streaming():
    """Show streaming dataset reading."""
    print_header("1. Dataset Streaming (Memory-Efficient)")
    
    data_path = "data/arxiv_2.9k.jsonl"
    print(f"Counting records in {data_path}...")
    
    count = 0
    for record in iter_jsonl(data_path):
        count += 1
        if count % 500 == 0:
            print(f"  Processed {count} records...")
    
    print(f"✅ Total records: {count:,}\n")
    print(f"Note: Records streamed one-by-one, not loaded to memory")


def demo_indexing():
    """Build or load vector index."""
    print_header("2. Vector Index (FAISS)")
    
    manager = IndexManager(data_path="data/arxiv_2.9k.jsonl")
    manager.initialize()
    
    return manager


def demo_search(manager):
    """Perform semantic search."""
    print_header("3. Semantic Search")
    
    query = "transformer models for natural language processing"
    results = manager.search_with_documents(query, k=3)
    
    print(f"\nTop 3 Results:\n")
    for i, (doc, distance) in enumerate(results, 1):
        similarity = max(0, 100 * (1 - distance / 10))
        print(f"{i}. {doc['title'][:60]}...")
        print(f"   ID: {doc['id']} | Similarity: {similarity:.1f}%")
        print(f"   Abstract: {doc['abstract'][:100]}...\n")


def demo_rag(manager):
    """Answer query using RAG."""
    print_header("4. RAG Query (Retrieval + LLM)")
    
    print("Loading LLM (TinyLlama)...")
    rag = RAGQuerySystem()
    
    # Use the SAME query as in demo_search for consistency
    query = "transformer models for natural language processing"
    print(f"Query: {query}\n")
    
    # Get relevant documents
    print("Retrieving relevant documents...")
    context_docs = manager.search_with_documents(query, k=3)
    
    print("\nRetrieved documents:")
    for i, (doc, dist) in enumerate(context_docs, 1):
        print(f"  {i}. {doc['title'][:70]}... (ID: {doc['id']})")
    
    # Generate answer
    print("\nGenerating answer with LLM...")
    result = rag.answer_query(query, context_docs)
    
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Sources ({result['num_sources']} documents):")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['title'][:60]}... (ID: {source['id']})")


def main():
    """Run complete demo."""
    print("\n" + "="*70)
    print(" " * 15 + "GenAI RAG Application Demo")
    print("="*70)
    print("\nMemory-Efficient Processing:")
    print("✓ Streaming: Read dataset line-by-line")
    print("✓ Batching: Build index in batches of 32")
    print("✓ On-Demand: Load only required documents")
    print("="*70)
    
    try:
        # 1. Show streaming
        demo_streaming()
        
        # 2. Build/load index
        manager = demo_indexing()
        
        # 3. Semantic search
        demo_search(manager)
        
        # 4. RAG query with LLM
        use_llm = os.getenv("SKIP_LLM", "false").lower() != "true"
        if use_llm:
            demo_rag(manager)
        else:
            print_header("4. RAG Query (Skipped)")
            print("LLM demo skipped (set SKIP_LLM=false to include it)")
            print("Note: LLM initialization takes ~30 seconds on first run\n")
        
        print("\n" + "="*70)
        print("✅ Demo Complete!")
        print("="*70)
        print("\nTo start the web server:")
        print("  python app.py")
        print("\nThen visit: http://127.0.0.1:8080")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
