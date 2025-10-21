"""
RAG Query System: Retrieval-Augmented Generation with local LLM
"""
from typing import List, Tuple, Iterator
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.generation.streamers import TextIteratorStreamer
import torch
from threading import Thread


class RAGQuerySystem:
    """
    Handles query answering using retrieved context + local LLM.
    Uses a lightweight model that can run on CPU.
    """
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize the LLM for text generation.
        
        Args:
            model_name: HuggingFace model to use (default: TinyLlama for fast CPU inference)
        """
        print(f"Loading LLM model: {model_name}...")
        
        # Use CPU if no GPU available
        device = 0 if torch.cuda.is_available() else -1
        
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float32 if device == -1 else torch.float16,
            device=device,
            max_new_tokens=512,
        )
        
        print(f"LLM loaded successfully on {'GPU' if device == 0 else 'CPU'}")
    
    def format_prompt(self, query: str, context_docs: List[Tuple[dict, float]]) -> str:
        """
        Format the prompt with retrieved context.
        
        Args:
            query: User's question
            context_docs: List of (document, distance) tuples from retrieval
            
        Returns:
            Formatted prompt string
        """
        # Build context from retrieved documents
        context_parts = []
        for i, (doc, distance) in enumerate(context_docs, 1):
            title = doc.get('title', 'Unknown')
            abstract = doc.get('abstract', '')
            doc_id = doc.get('id', 'unknown')
            
            context_parts.append(
                f"[Document {i} - ID: {doc_id}]\n"
                f"Title: {title}\n"
                f"Abstract: {abstract}\n"
            )
        
        context_text = "\n".join(context_parts)
        
        # Format prompt for chat model
        prompt = f"""<|system|>
You are a helpful AI assistant that answers questions based on provided research papers. Use only the information from the given documents to answer the question. If the documents don't contain relevant information, say so.
</s>
<|user|>
Context from research papers:

{context_text}

Question: {query}
</s>
<|assistant|>
"""
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer using the LLM.
        
        Args:
            prompt: Formatted prompt with context
            
        Returns:
            Generated answer
        """
        outputs = self.pipe(
            prompt,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
        )
        
        # Extract the generated text
        full_output = outputs[0]['generated_text']
        
        # Extract only the assistant's response
        if '<|assistant|>' in full_output:
            answer = full_output.split('<|assistant|>')[-1].strip()
        else:
            answer = full_output[len(prompt):].strip()
        
        # Clean up end tokens
        answer = answer.replace('</s>', '').strip()
        
        return answer
    
    def generate_answer_stream(self, prompt: str) -> Iterator[str]:
        """
        Generate answer using the LLM (streaming mode).
        
        Args:
            prompt: Formatted prompt with context
            
        Yields:
            Generated tokens one by one
        """
        streamer = TextIteratorStreamer(
            self.pipe.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = dict(
            text_inputs=prompt,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            max_new_tokens=512,
            streamer=streamer,
        )
        
        thread = Thread(target=self.pipe, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
    
    def answer_query(self, query: str, context_docs: List[Tuple[dict, float]], stream: bool = False):
        """
        Complete RAG pipeline: format prompt + generate answer.
        
        Args:
            query: User's question
            context_docs: Retrieved documents with distances
            stream: If True, returns generator; if False, returns dict
            
        Returns:
            Dictionary with answer OR generator for streaming
        """
        prompt = self.format_prompt(query, context_docs)
        
        # Format source documents with citations
        sources = []
        citations = []
        
        for doc, distance in context_docs:
            abstract = doc.get('abstract', '')
            citation = abstract[:200] + '...' if len(abstract) > 200 else abstract
            
            sources.append({
                'id': doc.get('id'),
                'title': doc.get('title'),
                'distance': float(distance)
            })
            
            citations.append({
                'doc_id': doc.get('id'),
                'title': doc.get('title'),
                'citation': citation
            })
        
        if stream:
            def stream_with_metadata():
                yield {
                    'type': 'metadata',
                    'query': query,
                    'sources': sources,
                    'citations': citations,
                    'num_sources': len(sources)
                }
                
                for token in self.generate_answer_stream(prompt):
                    yield {
                        'type': 'token',
                        'content': token
                    }
                
                yield {'type': 'done'}
            
            return stream_with_metadata()
        else:
            answer = self.generate_answer(prompt)
            
            return {
                'query': query,
                'answer': answer,
                'sources': sources,
                'citations': citations,
                'retrieved_context': '\n\n'.join([
                    f"• {c['title']}\n  {c['citation']}" 
                    for c in citations
                ]),
                'num_sources': len(sources)
            }
