from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import uuid

@dataclass
class DocumentChunk:
    # identity
    _id: str
    doc_id: str          # stable id per source file (filename hash)
    chunk_id: str        # per-chunk id
    source: str          # pdf|pptx|transcript|image|text

    # location metadata
    title: Optional[str]
    page: Optional[int] = None         # PDF page (1-based)
    slide: Optional[int] = None        # PPTX slide (1-based)
    timecode: Optional[str] = None     # e.g., "00:05:30-00:05:45" for video
    section: Optional[str] = None      # optional heading/shape label

    # text content
    text: str = ""                     # normalized chunk text
    caption: Optional[str] = None      # for images / slides OCR captions

    # aux
    confidence: Optional[float] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

    def to_es_doc(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_id", None)
        return d

def new_chunk(doc_id: str, source: str, **kwargs) -> DocumentChunk:
    return DocumentChunk(
        _id=str(uuid.uuid4()),
        doc_id=doc_id,
        chunk_id=str(uuid.uuid4()),
        source=source,
        **kwargs
    )
