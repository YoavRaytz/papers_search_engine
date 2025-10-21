from pathlib import Path
import json


def iter_jsonl(file_path, verbose=False):
    """
    Stream JSONL file line by line without loading everything into memory.
    Yields one record at a time.
    
    This is the ONLY way to read JSONL files - always uses streaming.
    
    Args:
        file_path: Path to JSONL file
        verbose: If True, print debug information
    
    Yields:
        Dictionary for each JSON record
    """
    records_read = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records_read += 1
                
                if verbose and records_read % 500 == 0:
                    print(f"   [Streaming] Read {records_read:,} records so far...")
                
                yield json.loads(line)
    
    if verbose:
        print(f"   [Streaming] Completed - read {records_read:,} total records")

