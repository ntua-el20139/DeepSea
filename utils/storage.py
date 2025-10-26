import os, json
from typing import List
from ingest.schema import DocumentChunk

CHUNK_DIR = os.path.join("data", "chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

def save_chunks(doc_id: str, chunks: List[DocumentChunk]):
    """Save chunks to data/chunks/<doc_id>.json for inspection/debugging."""
    path = os.path.join(CHUNK_DIR, f"{doc_id}.json")
    serializable = [c.to_es_doc() for c in chunks]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"[storage] saved {len(chunks)} chunks → {path}")
    return path
