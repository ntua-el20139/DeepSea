import threading
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Optional, Tuple
import numpy as np
from PIL import Image
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.rapid_ocr_model import RapidOcrModel

_MODEL_LOCK = threading.Lock()

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttf",
]


def _resolve_font_path() -> Optional[str]:
    """Return a local font path so RapidOCR avoids downloading assets."""
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def _load_model() -> RapidOcrModel:
    font_path = _resolve_font_path()
    options = RapidOcrOptions(
        force_full_page_ocr=True,
        font_path=font_path,
    )
    return RapidOcrModel(
        enabled=True,
        artifacts_path=None,
        options=options,
        accelerator_options=AcceleratorOptions(),
    )


def ocr_image(image: Image.Image) -> Tuple[str, Optional[float]]:
    """
    Run Docling's RapidOCR backend on a PIL image.
    Returns normalized text and an average confidence (0-100) if available.
    """
    arr = np.array(image.convert("RGB"))
    try:
        with _MODEL_LOCK:
            model = _load_model()
            result = model.reader(
                arr,
                use_det=model.options.use_det,
                use_cls=model.options.use_cls,
                use_rec=model.options.use_rec,
            )
    except Exception:
        return "", None

    texts = [t.strip() for t in (result.txts or []) if t and t.strip()] # type: ignore
    if not texts:
        return "", None

    confidence = None
    if result.scores: # type: ignore
        try:
            confidence = mean(result.scores) * 100.0 # type: ignore
        except Exception:
            confidence = None

    return "\n".join(texts), confidence
