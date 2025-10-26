import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from app.embedder import embed_texts
from elasticsearch import ApiError

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

def ensure_index():
    dim = len(embed_texts(["dimension probe"])[0])  # dummy probe to get dims
    try:
        if es.indices.exists(index=INDEX):
            print(f"[create_index] Index '{INDEX}' already exists")
            return
        body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index": {"knn": True},
            },
            "mappings": {
                "properties": {
                    "doc_id":   {"type": "keyword"},
                    "source":   {"type": "keyword"},
                    "title":    {"type": "text"},
                    "page":     {"type": "integer"},
                    "slide":    {"type": "integer"},
                    "timecode": {"type": "keyword"},
                    "section":  {"type": "keyword"},
                    "text":     {"type": "text"},
                    "caption":  {"type": "text"},
                    "confidence": {"type": "float"},
                    "created_at": {"type": "date"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": dim,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        es.indices.create(index=INDEX, body=body)
        print(f"[create_index] Created index '{INDEX}' with dims={dim}")
    except ApiError as err:
        print("[create_index] Elasticsearch returned:", err.status_code, err.body)
        raise

if __name__ == "__main__":
    ensure_index()
