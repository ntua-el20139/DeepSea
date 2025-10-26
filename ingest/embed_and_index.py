import os, uuid, time
from typing import Iterable, Dict, Any, List
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

from app.embedder import embed_texts
from ingest.pipeline import process_file
from utils.ids import stable_chunk_id
from ingest.create_index import ensure_index
from utils.storage import save_chunks
import glob


load_dotenv()
INDEX = str(os.getenv("INDEX_NAME"))
HOST = str(os.getenv("ES_HOST"))
USERNAME = str(os.getenv("ES_USERNAME"))
PASSWORD = str(os.getenv("ES_PASSWORD"))

es = Elasticsearch(
    hosts=[HOST],
    basic_auth=(USERNAME, PASSWORD),
    request_timeout=60,
    verify_certs=False,
    ssl_show_warn=False
)


def generate_actions(chunks: Iterable) -> Iterable[Dict[str, Any]]:
    """
    Takes DocumentChunk objects from process_file(), embeds text, and yields ES bulk actions.
    """
    buffer: List[Any] = []
    BATCH = 32

    def flush(buf):
        if not buf: return
        texts = [c.text for c in buf]
        vecs = embed_texts(texts)
        for c, v in zip(buf, vecs):
            doc = c.to_es_doc()
            doc["vector"] = v
            loc = str(c.page or c.slide or c.timecode or c.section or 0)
            yield {
                "_op_type": "update",
                "_index": INDEX,
                "_id": stable_chunk_id(c.doc_id, c.source, loc, c.text),
                "doc": doc,
                "doc_as_upsert": True
            }
        buf.clear()

    for c in chunks:
        buffer.append(c)
        if len(buffer) >= BATCH:
            yield from flush(buffer)
    # tail
    yield from flush(buffer)

def upload_files(paths: List[str], max_tokens=800, overlap_tokens=120):
    for path in paths:
        name = path.split("/")[-1].split(".")[0]
        print(f"[embed_and_index] Ingesting {path} ...")
        t0 = time.time()
        chunks = list(process_file(path, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
        print(f"[embed_and_index] {path}: {len(chunks)} unique chunks extracted")

        if chunks:
            save_chunks(name, chunks)

        print(f"[embed_and_index] Embedding & indexing ...")
        helpers.bulk(es.options(request_timeout=120), generate_actions(chunks), stats_only=True)
        print(f"[embed_and_index] Done {path} in {time.time() - t0:.2f}s")


# Insert all docs into ES index without the help of UI
if __name__ == "__main__":
    ensure_index()
    raw_files = glob.glob("data/raw/*")
    upload_files(raw_files, max_tokens=512, overlap_tokens=120)