import numpy as np
import faiss
from pathlib import Path
import pickle
import json

class VectorIndex:
    """
    Simple FAISS-based vector index for similarity search.
    """
    
    def __init__(self, dimension=384):
        """
        Args:
            dimension: Dimension of embedding vectors (e.g., 384 for sentence-transformers/all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.doc_ids = []  # Keep track of document IDs
        
    def add(self, embeddings, doc_ids):
        """
        Add vectors to the index.
        
        Args:
            embeddings: numpy array of shape (n, dimension)
            doc_ids: list of document IDs corresponding to embeddings
        """
        if len(embeddings) != len(doc_ids):
            raise ValueError("Number of embeddings must match number of doc_ids")
        
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.doc_ids.extend(doc_ids)
        
    def search(self, query_embedding, k=5):
        """
        Search for k nearest neighbors.
        
        Args:
            query_embedding: numpy array of shape (dimension,) or (1, dimension)
            k: number of results to return
            
        Returns:
            List of (doc_id, distance) tuples
        """
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.doc_ids):
                results.append((self.doc_ids[idx], float(dist)))
        return results
    
    def save(self, index_path, metadata_path):
        """
        Save index and metadata to disk.
        Supports both .pkl (old format) and .jsonl (new format) for metadata.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata (doc_ids)
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(index_path))
        
        # Check format based on file extension
        if str(metadata_path).endswith('.jsonl'):
            # New format: JSONL (streaming-friendly)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps({'dimension': self.dimension}) + '\n')
                for doc_id in self.doc_ids:
                    f.write(json.dumps({'doc_id': doc_id}) + '\n')
        else:
            # Old format: pickle (backward compatibility)
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'doc_ids': self.doc_ids,
                    'dimension': self.dimension
                }, f)
    
    @classmethod
    def load(cls, index_path, metadata_path):
        """
        Load index and metadata from disk.
        Supports both .pkl (old format) and .jsonl (new format).
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
            
        Returns:
            VectorIndex instance
        """
        index = faiss.read_index(str(index_path))
        metadata_path = Path(metadata_path)
        
        # Check format based on file extension
        if str(metadata_path).endswith('.jsonl'):
            # New format: JSONL
            doc_ids = []
            dimension = None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                metadata = json.loads(first_line)
                dimension = metadata['dimension']
                
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        doc_ids.append(record['doc_id'])
        else:
            # Old format: pickle (backward compatibility)
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                doc_ids = metadata['doc_ids']
                dimension = metadata['dimension']
        
        vector_index = cls(dimension=dimension)
        vector_index.index = index
        vector_index.doc_ids = doc_ids
        
        return vector_index
    
    def __len__(self):
        """Return number of vectors in the index."""
        return self.index.ntotal
