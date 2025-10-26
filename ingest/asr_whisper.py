# ingest/asr_whisper.py
from typing import Dict, List, Any, Optional
import os
from faster_whisper import WhisperModel

# Config via env (sane defaults)
_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")        # tiny/base/small/medium/large-v2
_WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")       # "cpu", "cuda", "auto"
_WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "auto")     # "auto","int8","int8_float16","float16","float32"
_WHISPER_BEAM = int(os.getenv("WHISPER_BEAM", "1"))         # 1 = greedy, >1 = beam search
_WHISPER_LANG = os.getenv("WHISPER_LANG", "en")             # e.g. "en", "el" (Greek), or None for auto

def _load_model() -> WhisperModel:
    print(f"[asr_whisper] loading model '{_WHISPER_MODEL}' on device={_WHISPER_DEVICE}, compute_type={_WHISPER_COMPUTE}")
    return WhisperModel(_WHISPER_MODEL, device=_WHISPER_DEVICE, compute_type=_WHISPER_COMPUTE)

def transcribe_whisper(
    media_path: str,
    language: Optional[str] = _WHISPER_LANG,
    vad_filter: bool = True,
    temperature: float = 0.0,
    beam_size: int = _WHISPER_BEAM
) -> Dict[str, Any]:
    """
    Transcribe an audio/video file using faster-whisper.
    Returns: {"segments":[{"text","start","end"}...], "language": "<detected or provided>"}
    """
    model = _load_model()
    print(f"[asr_whisper] transcribing file: {media_path}")

    # Note: faster-whisper decodes audio internally; no temp files needed.
    segments, info = model.transcribe(
        media_path,
        language=language,                      # None => auto-detect
        vad_filter=vad_filter,
        beam_size=beam_size,
        temperature=temperature,
        condition_on_previous_text=True,
        word_timestamps=False                   # we use segment timestamps; enable True if you need words
    )

    out_segments: List[Dict[str, Any]] = []
    count = 0
    for seg in segments:
        count += 1
        out_segments.append({
            "text": seg.text.strip(),
            "start": float(seg.start),
            "end": float(seg.end)
        })
    detected_lang = language or getattr(info, "language", None)
    print(f"[asr_whisper] done: {count} segments, language={detected_lang}")
    return {"segments": out_segments, "language": detected_lang}
