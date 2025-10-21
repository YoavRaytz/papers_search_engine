# Datasets Module Documentation

## Overview

The `datasets` module is responsible for **memory-efficient streaming processing** of JSONL (JSON Lines) dataset files. This module is designed to handle datasets of any size without loading them entirely into memory, making it scalable to millions of records.

---

## Module Structure

```
datasets/
├── __init__.py
└── jsonl_dataset.py    # Core streaming functionality
```

---

## jsonl_dataset.py - Streaming JSONL Reader

### Purpose

Provides a **streaming iterator** for reading JSONL files line-by-line without loading the entire file into memory. This is the **only way** JSONL files are read throughout the application.

### Key Concepts

- **Streaming Processing**: Reads one record at a time using Python generators
- **Memory Efficiency**: Only one line is in memory at any given time
- **Large File Support**: Can handle datasets larger than available RAM
- **Real-time Progress**: Optional verbose mode for monitoring progress

---

### Core Function: `iter_jsonl()`

```python
def iter_jsonl(file_path, verbose=False):
    """
    Stream JSONL file line by line without loading everything into memory.
    Yields one record at a time.
    
    Args:
        file_path: Path to JSONL file
        verbose: If True, print debug information
    
    Yields:
        Dictionary for each JSON record
    """
```

**Key Characteristics:**
- Opens file in UTF-8 text mode
- Reads line-by-line using Python's file iterator (memory-efficient)
- Skips empty lines
- Parses each line as JSON using `json.loads()`
- Uses `yield` - returns one record at a time (generator pattern)
- Peak memory: ~1KB per record (one line in memory)
- No file size limit (can process files larger than RAM)

---

## Integration with Other Modules

### Used by Index Manager

The `index` module uses streaming for building vector indices:

```python
# index_manager.py
from ..datasets.jsonl_dataset import iter_jsonl

def build_index(self):
    # Stream through dataset to build index
    for record in iter_jsonl(self.data_path):
        text = f"{record['title']} {record['abstract']}"
        embedding = self.model.encode([text])[0]
        self.vector_index.add(embedding, record['id'])
        
        # Metadata written immediately to disk (not kept in RAM)
        write_metadata(record)
```

### Used for Document Retrieval

Loading specific documents after search:

```python
# index_manager.py
def search_with_documents(self, query, k=5):
    # Get IDs from vector search
    result_ids = self.search(query, k=k)
    
    # Stream dataset to find matching documents
    docs = {}
    for record in iter_jsonl(self.data_path):
        if record['id'] in result_ids:
            docs[record['id']] = record
            if len(docs) == k:
                break  # Found all, stop streaming
    
    return docs
```

---

## Summary

The datasets module provides the foundation for memory-efficient data processing in the GenAI application:

- ✅ **Streaming-first design** - Never loads entire dataset into memory
- ✅ **Scalable** - Handles datasets from KB to TB
- ✅ **Simple API** - Single iterator function for all JSONL reading
- ✅ **Composable** - Easy to chain with other processing steps
- ✅ **Reliable** - Predictable memory usage and performance

This design enables the application to work with large academic paper datasets (millions of papers) while maintaining low memory footprint and fast startup times.
