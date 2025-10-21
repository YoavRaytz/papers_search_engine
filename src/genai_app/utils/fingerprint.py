import hashlib
from pathlib import Path

def file_fingerprint(file_path):
    """
    Calculate SHA256 hash of file to detect changes.
    Returns hex string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
