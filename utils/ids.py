import hashlib

def file_sha256(path: str, chunk_size: int = 1_048_576) -> str:
    print(f"[ids] hashing file for doc_id: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    digest = h.hexdigest()
    print(f"[ids] sha256: {digest}")
    return digest

def stable_chunk_id(doc_id: str, source: str, loc: str, text: str) -> str:
    sig = hashlib.sha1((loc + "|" + text).encode("utf-8")).hexdigest()[:12]
    cid = f"{doc_id}:{source}:{loc}:{sig}"
    return cid
