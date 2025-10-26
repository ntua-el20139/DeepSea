import os
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

from app.embedder import embed_texts

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

def _knn(query_vec, k=20, num_candidates=200, source=None):
    return es.search(
        index=INDEX,
        knn={
            "field": "vector",
            "query_vector": query_vec,
            "k": k,
            "num_candidates": max(num_candidates, k*10),
        },
        size=k,
        source={"includes": source} if source is not None else {"includes": ["text","title","page","slide","uri","source","caption","extra"]},
    )

def _bm25(query: str, size=20, source=None):
    return es.search(
        index=INDEX,
        query={
            "multi_match": {
                "query": query,
                "fields": ["text^3", "title^2", "caption"],
                "operator": "and"
            }
        },
        size=size,
        highlight={"fields": {"text": {"fragment_size": 180, "number_of_fragments": 1}}},
        source={"includes": source} if source is not None else {"includes": ["text","title","page","slide","uri","source","caption","extra"]},
    )

def hybrid_search(query: str, k: int = 6, min_score: float = 0.03) -> List[Dict[str, Any]]:
    qvec = embed_texts([query])[0]

    vres = _knn(qvec, k=20)
    bres = _bm25(query, size=20)

    # Reciprocal Rank Fusion (simple, strong baseline)
    def rrf(hits, K=60):
        return {h["_id"]: 1.0 / (K + r) for r, h in enumerate(hits, 1)}

    sv = rrf(vres["hits"]["hits"])
    sb = rrf(bres["hits"]["hits"])

    # merge scores
    scores = {}
    by_id = {}
    for h in vres["hits"]["hits"] + bres["hits"]["hits"]:
        _id = h["_id"]
        by_id[_id] = h
        scores[_id] = scores.get(_id, 0.0) + (sv.get(_id, 0.0) + sb.get(_id, 0.0))

    ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("Ranked IDs and scores:")
    for _id, score in ranked_ids:
        print(f"ID: {_id}, Score: {score}")
    filtered_ids = [(_id, score) for _id, score in ranked_ids if score > min_score][:k]
    print("Filtered IDs after min_score:")
    for _id, score in filtered_ids:
        print(f"ID: {_id}, Score: {score}")

    results = []
    per_doc_counts = {}
    for _id, _ in filtered_ids:
        h = by_id[_id]
        s = h["_source"]
        doc_id = _id.split(":")[0]
        
        # limit to 2 results per document to increase source diversity
        if doc_id:
            count = per_doc_counts.get(doc_id, 0)
            if count >= 2:
                continue
            per_doc_counts[doc_id] = count + 1
        
        # attach highlight snippet if available
        snippet = None
        try:
            snippet = h.get("highlight", {}).get("text", [None])[0]
        except Exception:
            pass
        s["_id"] = _id
        s["_score"] = scores[_id]
        if snippet: s["snippet"] = snippet
        results.append(s)
    return results
