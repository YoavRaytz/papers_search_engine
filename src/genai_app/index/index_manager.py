from pathlib import Path
import json
import hashlib
from sentence_transformers import SentenceTransformer
from ..datasets.jsonl_dataset import iter_jsonl
from ..utils.fingerprint import file_fingerprint
from .vector_index import VectorIndex

class IndexManager:
    """
    Manages the lifecycle of the vector index:
    - Build index from dataset using streaming (memory-efficient)
    - Save/load index to/from disk
    - Detect dataset changes and rebuild when needed
    - Does NOT store all documents in memory (scalable to millions of records)
    """
    
    def __init__(self, data_path, index_dir=".index", model_name="all-MiniLM-L6-v2"):
        """
        Args:
            data_path: Path to JSONL dataset file
            index_dir: Directory to store index files
            model_name: Name of sentence-transformers model to use
        """
        self.data_path = Path(data_path)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.index_dir / "faiss.index"
        self.metadata_file = self.index_dir / "metadata.jsonl"  # Changed to JSONL for streaming
        self.fingerprint_file = self.index_dir / "dataset_fingerprint.txt"
        
        print(f"[IndexManager] Initializing with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.vector_index = None
        self.num_documents = 0
        
    def _get_current_fingerprint(self):
        """Get fingerprint of current dataset file."""
        return file_fingerprint(self.data_path)
    
    def _get_saved_fingerprint(self):
        """Get saved fingerprint from last index build."""
        if not self.fingerprint_file.exists():
            return None
        return self.fingerprint_file.read_text().strip()
    
    def _save_fingerprint(self, fingerprint):
        """Save dataset fingerprint."""
        self.fingerprint_file.write_text(fingerprint)
    
    def _should_rebuild(self):
        """Check if index needs to be rebuilt."""
        # No index exists
        if not self.index_file.exists() or not self.metadata_file.exists():
            return True
        
        # Dataset changed
        current_fp = self._get_current_fingerprint()
        saved_fp = self._get_saved_fingerprint()
        
        return current_fp != saved_fp
    
    def build_index(self, batch_size=32):
        """
        Build vector index from dataset using streaming in batches.
        Saves metadata to disk - does NOT keep all documents in memory!
        
        Args:
            batch_size: Number of records to process at once for embedding
        """
        print(f"\n{'='*70}")
        print(f"Building Vector Index")
        print(f"{'='*70}")
        print(f"Dataset: {self.data_path}")
        print(f"Batch size: {batch_size} | Mode: Streaming (memory-efficient)")
        
        # Get embedding dimension from model
        dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {dimension}")
        self.vector_index = VectorIndex(dimension=dimension)
        
        batch_texts = []
        batch_ids = []
        total_records = 0
        batch_count = 0
        
        # Open metadata file for writing (streaming!)
        with open(self.metadata_file, 'w', encoding='utf-8') as meta_file:
            
            print(f"Processing dataset in batches...")
            
            # Stream through dataset
            for record in iter_jsonl(self.data_path):
                # Write metadata immediately (not stored in RAM!)
                meta_record = {
                    'idx': total_records,
                    'id': record['id'],
                    'title': record['title']
                }
                meta_file.write(json.dumps(meta_record, ensure_ascii=False) + '\n')
                
                # Combine title and abstract for embedding
                text = f"{record['title']} {record['abstract']}"
                batch_texts.append(text)
                batch_ids.append(record['id'])
                
                # Process batch when full
                if len(batch_texts) >= batch_size:
                    batch_count += 1
                    embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                    self.vector_index.add(embeddings, batch_ids)
                    total_records += len(batch_texts)
                    
                    # Progress indicator every 10 batches
                    if batch_count % 10 == 0:
                        print(f"  Processed {total_records:,} documents ({batch_count} batches)...")
                    
                    batch_texts = []
                    batch_ids = []
            
            # Process remaining records
            if batch_texts:
                batch_count += 1
                embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                self.vector_index.add(embeddings, batch_ids)
                total_records += len(batch_texts)
        
        self.num_documents = total_records
        
        print(f"\n✅ Index built successfully")
        print(f"   Documents: {total_records:,} | Batches: {batch_count} | Peak RAM: ~{batch_size} docs")
        print(f"   Metadata saved to: {self.metadata_file}")
        print(f"{'='*70}\n")
        
        # Save everything
        self.save_index()
    
    def save_index(self):
        """Save index and fingerprint to disk."""
        if self.vector_index is None:
            raise ValueError("No index to save. Build index first.")
        
        print(f"Saving index to {self.index_dir}...")
        self.vector_index.save(self.index_file, self.metadata_file)
        
        fingerprint = self._get_current_fingerprint()
        self._save_fingerprint(fingerprint)
        
        print("Index saved successfully")
    
    def load_index(self):
        """Load existing index from disk."""
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_file}")
        
        print(f"\nLoading index from {self.index_dir}...")
        self.vector_index = VectorIndex.load(self.index_file, self.metadata_file)
        self.num_documents = len(self.vector_index)
        print(f"✅ Loaded {self.num_documents:,} vectors (metadata on disk)\n")
    
    def initialize(self, force_rebuild=False):
        """
        Initialize the index manager:
        - Load existing index if available and dataset hasn't changed
        - Build new index if needed
        
        Args:
            force_rebuild: Force rebuild even if index exists
        """
        if force_rebuild or self._should_rebuild():
            print("Index needs to be rebuilt")
            self.build_index()
        else:
            print("Loading existing index")
            self.load_index()
    
    def search(self, query, k=5):
        """
        Search for similar documents.
        Searches ALL vectors in FAISS index (not in batches/chunks).
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (doc_id, distance) tuples
        """
        if self.vector_index is None:
            raise ValueError("Index not initialized. Call initialize() first.")
        
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Search ALL vectors in the index!
        results = self.vector_index.search(query_embedding, k=k)
        
        return results
    
    def search_with_documents(self, query, k=5):
        """
        Search for similar documents and return full document content.
        Loads only the k documents we need - NOT the entire dataset!
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (document_dict, distance) tuples
        """
        # Get doc IDs and distances
        print(f"Searching for: '{query}'")
        results = self.search(query, k=k)
        print(f"Found {len(results)} matches, loading full documents...")
        
        # Load full documents from dataset (streaming!)
        doc_id_to_doc = {}
        records_scanned = 0
        
        for record in iter_jsonl(self.data_path):
            records_scanned += 1
            
            if record['id'] in [doc_id for doc_id, _ in results]:
                doc_id_to_doc[record['id']] = record
                
                if len(doc_id_to_doc) == len(results):
                    print(f"Loaded {k} documents (scanned {records_scanned}/{self.num_documents:,})")
                    break
        
        # Return documents in same order as results
        results_with_docs = []
        for doc_id, distance in results:
            if doc_id in doc_id_to_doc:
                results_with_docs.append((doc_id_to_doc[doc_id], distance))
        
        return results_with_docs
