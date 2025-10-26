from typing import Dict, Any
from app.retrieve import hybrid_search
from app.answer import answer_query

def ask(query: str, k: int = 6, max_tokens=350, temperature=0.05) -> Dict[str, Any]:
    print(f"[qa] Received query: {query}")
    hits = hybrid_search(query, k=k)
    answer = answer_query(query, hits, max_tokens=max_tokens, temperature=temperature)

    sources = [
        {
            "title": h.get("title"),
            "page": h.get("page"),
            "slide": h.get("slide"),
            "uri": h.get("uri"),
            "snippet": h.get("snippet", h.get("text","")[:220] + "…"),
            "source": h.get("source")
        }
        for h in hits
    ]
    print(f"[qa] Generated answer: {answer}")
    for s in sources:
        print(f"[qa] Source: {s['title']} - {s.get('page', s.get('slide',''))} - {s['source']}")
    return {"answer": answer, "sources": sources}
