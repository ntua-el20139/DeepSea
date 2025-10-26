import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

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

def clear_index(index_name: str):
    """
    Deletes all documents in the specified index while keeping the index itself.
    """
    if not es.indices.exists(index=index_name):
        print(f"Index '{index_name}' does not exist.")
        return

    print(f"Deleting all documents from index '{index_name}'...")
    es.delete_by_query(
        index=index_name,
        body={"query": {"match_all": {}}},
        refresh=True,  # Make deletion visible immediately
        conflicts="proceed",
    )
    print(f"All documents deleted from index '{index_name}'.")

if __name__ == "__main__":
    clear_index(INDEX)