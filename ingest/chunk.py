import math
from typing import List

import nltk

try:
    import tiktoken

    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENCODER = None

_TOKEN_HEADROOM = 64

print("[chunk] module loaded; ensuring punkt...")
nltk.download("punkt")
_SENT_SPLIT = nltk.data.load("tokenizers/punkt/english.pickle")

def approx_tokens(s: str) -> int:
    if _TIKTOKEN_ENCODER is not None:
        return max(1, len(_TIKTOKEN_ENCODER.encode(s)))
    # Fallback heuristic errs on the conservative side if tiktoken is unavailable.
    return max(1, math.ceil(len(s) / 3))

def split_into_sentences(text: str) -> List[str]:
    try:
        sents = _SENT_SPLIT.tokenize(text)  # type: ignore
        if sents:
            return sents
    except Exception:
        pass
    return [text]

def _split_long_sentence(s: str, max_chars: int) -> List[str]:
    words = s.split()
    if not words:
        return [s.strip()] if s.strip() else []

    parts: List[str] = []
    cur_words: List[str] = []
    cur_len = 0

    for word in words:
        addition = len(word) + (1 if cur_words else 0)
        if cur_len + addition > max_chars and cur_words:
            parts.append(" ".join(cur_words).strip())
            cur_words = [word]
            cur_len = len(word)
        else:
            cur_words.append(word)
            cur_len += addition

    if cur_words:
        parts.append(" ".join(cur_words).strip())
    return [p for p in parts if p]

def _enforce_token_cap(text: str, max_tokens: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if approx_tokens(text) <= max_tokens:
        return [text]

    mid = len(text) // 2
    split = text.rfind(" ", 0, mid)
    if split == -1:
        split = text.find(" ", mid)

    if split <= 0 or split >= len(text):
        split = min(len(text), max_tokens * 4)
        if split <= 0 or split >= len(text):
            return [text]

    left = text[:split].strip()
    right = text[split:].strip()
    parts: List[str] = []
    parts.extend(_enforce_token_cap(left, max_tokens))
    parts.extend(_enforce_token_cap(right, max_tokens))
    return [p for p in parts if p]

def chunk_by_tokens(text: str, max_tokens: int = 512, overlap_tokens: int = 120) -> List[str]:
    if not text.strip():
        return []
    effective_max = max(1, max_tokens - _TOKEN_HEADROOM)
    sents = split_into_sentences(text)
    chunks, cur, cur_tokens = [], [], 0

    for s in sents:
        st = approx_tokens(s)
        if cur_tokens + st <= effective_max:
            cur.append(s); cur_tokens += st
        else:
            if cur:
                chunks.append(" ".join(cur).strip())
                # build overlap
                tail, tail_tokens = [], 0
                for ss in reversed(cur):
                    t = approx_tokens(ss)
                    if tail_tokens + t <= overlap_tokens:
                        tail.insert(0, ss); tail_tokens += t
                    else:
                        break
                cur = tail + [s]
                cur_tokens = sum(approx_tokens(x) for x in cur)
            else:
                # a single huge sentence — split more nicely
                for sub in _split_long_sentence(s, effective_max * 4):
                    if sub:
                        chunks.append(sub)
                cur, cur_tokens = [], 0

    if cur:
        chunks.append(" ".join(cur).strip())

    bounded: List[str] = []
    for ch in chunks:
        bounded.extend(_enforce_token_cap(ch, effective_max))
    chunks = [c.strip() for c in bounded if c.strip()]

    print(f"[chunk] produced {len(chunks)} chunks (max_tokens={max_tokens}, effective_max={effective_max}, overlap_tokens={overlap_tokens})")
    return chunks
