import os, requests
from typing import List, Dict, Any
from ingest.watsonx import get_watsonx_headers


WATSONX_BASE_URL = os.getenv("WATSONX_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
GEN_URL = f"{WATSONX_BASE_URL}/ml/v1/text/chat?version=2024-10-08"

SYSTEM = """You are a retrieval-augmented assistant for DeepSea.
Your task is to answer user questions on the Pythia platform based on the provided context from relevant documents.

Rules (follow all):
1) Base your answer primarily on facts found in the Context. You may perform light reasoning or summarization to clarify, connect, or interpret these facts — but do NOT introduce new information not supported by the Context.
2) If the Context does not contain enough information to answer confidently, reply exactly: "I can't answer that based on the provided documents."
3) Answer in a markdown format, using lists for multiple items when appropriate.
4) Preserve exact strings for emails, phone numbers, URLs, IDs, units and amounts (do not reformat or invent).
5) Be concise and also polite. Provide ONLY the answer — no speculation, no filler, and no references to the Context or user prompt.
6) Respond in the same language as the user's question.
7) Make logical steps so the answer is more complete (e.g., if the question is "Is the Pythia login page secure?", and the context says "The Pythia login page uses HTTPS", you can answer "Yes, the Pythia login page is secure because it uses HTTPS").
"""

def _fmt_tag(r: Dict[str, Any]) -> str:
    if r.get("page"):  return f"{r.get('title','')}, p.{r['page']}"
    if r.get("slide"): return f"{r.get('title','')}, slide {r['slide']}"
    return r.get("title","source")

def _format_context(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for r in results:
        tag = _fmt_tag(r)
        txt = r.get("snippet") or r.get("text","")
        blocks.append(f"[{tag}] {txt}")
    return "\n\n".join(blocks)

def answer_query(query: str, results: List[Dict[str, Any]], max_tokens=350, temperature=0.0, seed: int = 1337) -> str:
    context = _format_context(results)
    headers = get_watsonx_headers()
    payload = {
        "model_id": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Context:\n{context}\n\nUser: {query}\nAnswer:"
                    }
                ]
            }
        ],
        "decoding_method": "greedy",
        "max_tokens": max_tokens,
        "repetition_penalty": 1.05,
        "temperature": temperature,
        "top_p": 0.1,
        "seed": seed,
        "project_id": PROJECT_ID
    }
    r = requests.post(GEN_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
