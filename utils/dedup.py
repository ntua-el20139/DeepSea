import hashlib, re
from collections import Counter
from typing import List, Set

_WS = re.compile(r"[ \t]+")
_PAGENO = re.compile(r"^\s*(page\s*\d+|\d+)\s*$", re.IGNORECASE)

def canonicalize_for_hash(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in s.split("\n"):
        L = line.strip()
        if not L or _PAGENO.match(L):
            continue
        L = L[:-1] if L.endswith(".") else L
        lines.append(L)
    s = ". ".join(lines)
    s = _WS.sub(" ", s).strip().lower()
    return s

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def find_boilerplate_lines(pages_text: List[str], min_frac: float = 0.6, max_len: int = 120) -> Set[str]:
    print(f"[dedup] scanning {len(pages_text)} pages/slides for boilerplate (min_frac={min_frac})")
    total = max(len(pages_text), 1)
    counts = Counter()
    for t in pages_text:
        seen = set()
        for line in (t or "").split("\n"):
            L = line.strip()
            if 0 < len(L) <= max_len:
                seen.add(L)
        counts.update(seen)
    boiler = {line for line, c in counts.items() if c / total >= min_frac}
    print(f"[dedup] found {len(boiler)} boilerplate lines")
    return boiler

def drop_boilerplate(s: str, boiler: Set[str]) -> str:
    if not boiler:
        return s or ""
    kept = [line for line in (s or "").split("\n") if line.strip() and line.strip() not in boiler]
    return "\n".join(kept)

def is_duplicate(hash_val: str, seen: Set[str]) -> bool:
    if hash_val in seen:
        return True
    seen.add(hash_val)
    return False
