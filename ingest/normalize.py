import re

print("[normalize] module loaded")

_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"-\n", "", s)       # join hyphenated line breaks
    s = _NL.sub("\n\n", s)          # compress tall gaps
    s = _WS.sub(" ", s)             # collapse spaces/tabs
    s = re.sub(r"(?m)^[•▪\-]+\s*", "", s)  # strip leading bullets
    return s.strip()
