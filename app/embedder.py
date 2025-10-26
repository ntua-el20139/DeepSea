import os
import requests
from typing import List
from dotenv import load_dotenv
from ingest.watsonx import get_watsonx_headers

load_dotenv()

WATSONX_BASE_URL = os.getenv("WATSONX_BASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
API_KEY = str(os.getenv("WATSONX_API_KEY"))
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
EMBED_URL = f"{WATSONX_BASE_URL}/ml/v1/text/embeddings?version=2023-10-25"


def embed_texts(texts: List[str]) -> List[list]:
    """
    Returns a list of embedding vectors (list[float]) in the same order.
    Sends in small batches to avoid payload limits.
    """
    out: List[list] = []
    BATCH = 16

    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        payload = {
            "model_id": EMBED_MODEL,
            "inputs": batch,
            "project_id": PROJECT_ID,
        }
        r = requests.post(EMBED_URL, headers=get_watsonx_headers(), json=payload, timeout=60)
        if not r.ok:
            print("Watsonx error:", r.status_code, r.text)
            r.raise_for_status()

        data = r.json()
        out.extend([row["embedding"] for row in data["results"]])
    return out
