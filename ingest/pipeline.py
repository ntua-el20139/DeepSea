import math
import subprocess
import tempfile
from typing import Iterator, Set, List, Tuple, Optional, Dict
from pathlib import Path
from .schema import new_chunk, DocumentChunk
from .normalize import normalize_text
from .chunk import chunk_by_tokens
from .loaders import (
    load_pdf_text,
    load_pdf_images_ocr,
    load_pdf_tables,
    load_pptx,
    ocr_images,
    load_plain_text,
    load_docx_blocks,
)
from utils.ids import file_sha256
from utils.dedup import canonicalize_for_hash, sha1_str, find_boilerplate_lines, drop_boilerplate, is_duplicate
from .asr_whisper import transcribe_whisper
from .asr_segments import segments_to_blocks

_VIDEO_SEGMENT_LIMIT_MB = 100
_VIDEO_SEGMENT_LIMIT_BYTES = _VIDEO_SEGMENT_LIMIT_MB * 1024 * 1024
SUPPORTED = {".pdf", ".pptx", ".txt", ".md", ".docx", ".mp4"}

print("[pipeline] module loaded")


def guess_title_from_path(p: Path) -> str:
    return p.stem.replace("_", " ").replace("-", " ").strip().title()


def doc_id_from_path(p: Path) -> str:
    return file_sha256(str(p))[:32]


def process_pdf(path: Path, max_tokens=512, overlap_tokens=120) -> Iterator[DocumentChunk]:
    doc_id = doc_id_from_path(path)
    title = guess_title_from_path(path)
    print(f"[pipeline/pdf] start: {path}, doc_id={doc_id}")

    # 1) First pass for boilerplate detection (native text only)
    native_pages = [raw or "" for _, raw in load_pdf_text(str(path))]
    boiler = find_boilerplate_lines(native_pages, min_frac=0.6)
    tables_by_page = load_pdf_tables(str(path))
    ocr_iter = iter(load_pdf_images_ocr(str(path)))
    ocr_cache: Dict[int, Tuple[str, Optional[float]]] = {}

    def get_ocr(page_no: int) -> Optional[Tuple[str, Optional[float]]]:
        if page_no in ocr_cache:
            return ocr_cache[page_no]
        for p, text, conf in ocr_iter:
            ocr_cache[p] = (text, conf)
            if p == page_no:
                return ocr_cache[p]
        return None

    seen_page_hashes: Set[str] = set()
    # 2) Loop again: selective OCR + dedup + chunk
    for page, raw in load_pdf_text(str(path)):
        caption = None
        confidence = None
        base = normalize_text(drop_boilerplate(raw or "", boiler))

        if len(base.split()) < 10:
            # OCR this page only
            ocr_result = get_ocr(page)
            if ocr_result:
                ocr_text, ocr_conf = ocr_result
                ocr_clean = normalize_text(drop_boilerplate(ocr_text or "", boiler))
                if len(ocr_clean.split()) > len(base.split()):
                    base = ocr_clean
                    caption = "ocr"
                    confidence = ocr_conf

        canon = canonicalize_for_hash(base)
        if not canon:
            continue
        if is_duplicate(sha1_str(canon), seen_page_hashes):
            print(f"[pipeline/pdf] skip duplicate page {page}")
            continue

        if len(base.split()) >= 8:
            chunks = chunk_by_tokens(base, max_tokens, overlap_tokens)
            print(f"[pipeline/pdf] page {page} -> {len(chunks)} chunks (caption={caption})")
            for ch in chunks:
                yield new_chunk(doc_id=doc_id, source="pdf",
                                title=title, page=page, text=ch, caption=caption,
                                confidence=confidence)

        # Table chunks
        table_blocks = tables_by_page.get(page)
        if table_blocks:
            for tbl in table_blocks:
                tbl_norm = normalize_text(tbl)
                if len(tbl_norm.split()) < 4:
                    continue
                tchunks = chunk_by_tokens(tbl_norm, max_tokens, overlap_tokens)
                print(f"[pipeline/pdf] page {page} table -> {len(tchunks)} chunks")
                for ch in tchunks:
                    yield new_chunk(
                        doc_id=doc_id,
                        source="pdf",
                        title=title,
                        page=page,
                        text=ch,
                        caption="table",
                    )

def process_pptx(path: Path, max_tokens=512, overlap_tokens=120) -> Iterator[DocumentChunk]:
    doc_id = doc_id_from_path(path)
    title = guess_title_from_path(path)
    print(f"[pipeline/pptx] start: {path}, doc_id={doc_id}")

    slides = [(sn, txt or "", imgs, tables) for sn, txt, imgs, tables in load_pptx(str(path))]
    boiler = find_boilerplate_lines([t for _, t, _, _ in slides], min_frac=0.7)

    seen_slide_hashes: Set[str] = set()
    for slide_no, txt, imgs, tables in slides:
        slide_text = normalize_text(drop_boilerplate(txt, boiler))
        table_md = "\n\n".join(tables) if tables else ""
        ocr_txt = ""
        ocr_conf = None
        if imgs:
            big = [im for im in imgs if im.width * im.height > 150_000]
            if big:
                raw_txt, ocr_conf = ocr_images(big)
                ocr_txt = normalize_text(raw_txt)

        canon = canonicalize_for_hash(slide_text + ("\n" + table_md if table_md else "") + ("\n" + ocr_txt if ocr_txt else ""))
        if canon and is_duplicate(sha1_str(canon), seen_slide_hashes):
            print(f"[pipeline/pptx] skip duplicate slide {slide_no}")
            continue

        # Slide text chunks
        if len(slide_text.split()) >= 4:
            chunks = chunk_by_tokens(slide_text, max_tokens, overlap_tokens)
            print(f"[pipeline/pptx] slide {slide_no} text -> {len(chunks)} chunks")
            for ch in chunks:
                yield new_chunk(doc_id=doc_id, source="pptx", title=title, slide=slide_no, text=ch)
        # Table chunks
        if table_md:
            tchunks = chunk_by_tokens(table_md, max_tokens, overlap_tokens)
            print(f"[pipeline/pptx] slide {slide_no} tables -> {len(tchunks)} chunks")
            for ch in tchunks:
                yield new_chunk(doc_id=doc_id, source="pptx", title=title, slide=slide_no, text=ch, caption="table")
        # OCR chunks
        if ocr_txt and len(ocr_txt.split()) >= 8:
            ochunks = chunk_by_tokens(ocr_txt, max_tokens, overlap_tokens)
            print(f"[pipeline/pptx] slide {slide_no} ocr -> {len(ochunks)} chunks")
            for ch in ochunks:
                yield new_chunk(doc_id=doc_id, source="pptx", title=title, slide=slide_no,
                                text=ch, caption="ocr", confidence=ocr_conf)


def process_docx(path: Path, max_tokens=512, overlap_tokens=80) -> Iterator[DocumentChunk]:
    doc_id = doc_id_from_path(path)
    title = guess_title_from_path(path)
    print(f"[pipeline/docx] start: {path}, doc_id={doc_id}")

    # --- Load blocks and initialize state
    blocks = list(load_docx_blocks(str(path)))
    print(f"[pipeline/docx] {len(blocks)} blocks loaded from docx")

    # Buffers and state tracking
    buf: list[str] = []
    previous_section: str = ""
    num_blocks: int = 0
    sections = {"bigger_font", "underline", "bold", "italic", "paragraph_break"}
    MIN_BLOCKS_FOR_SECTION = 3

    def flush_buf() -> list[str]:
        """Normalize and chunk accumulated text, then clear buffer."""
        nonlocal buf, num_blocks
        if not buf:
            return []
        text = normalize_text("\n".join(buf))
        chunks = chunk_by_tokens(text, max_tokens, overlap_tokens)
        buf = []
        num_blocks = 0  # reset counter after flush
        return chunks

    # --- Iterate through all parsed document blocks
    for b in blocks:
        current_text = b.get("text", "")
        current_section = b["type"]

        # Warm-up period before trusting section boundaries
        if num_blocks < MIN_BLOCKS_FOR_SECTION:
            if num_blocks == 0:
                previous_section = current_section
            if current_text:
                buf.append(current_text)
                num_blocks += 1
            continue

        # Section boundary — new heading or formatting change
        if (current_section == "heading" and b.get("level", 6) <= 3) or current_section in sections:
            chunks = flush_buf()
            if chunks:
                print(f"[pipeline/docx] section '{previous_section}' -> {len(chunks)} chunks")
                for ch in chunks:
                    yield new_chunk(
                        doc_id=doc_id,
                        source="docx",
                        title=title,
                        section=previous_section,
                        text=ch,
                    )
            previous_section = current_section
            if current_text:
                buf.append(current_text)
                num_blocks += 1
            continue

        # Table block → markdown, chunk, yield
        if current_section == "table":
            chunks = flush_buf()
            if chunks:
                print(f"[pipeline/docx] section '{previous_section}' -> {len(chunks)} chunks")
                for ch in chunks:
                    yield new_chunk(doc_id=doc_id, source="docx", title=title,
                                    section=previous_section, text=ch)
            md = normalize_text(b["as_markdown"])
            tchunks = chunk_by_tokens(md, max_tokens, overlap_tokens)
            print(f"[pipeline/docx] table -> {len(tchunks)} chunks")
            for ch in tchunks:
                yield new_chunk(doc_id=doc_id, source="docx", title=title,
                                section=current_section, text=ch, caption="table")
            num_blocks = 0
            continue

        # Image block → OCR if large enough
        if current_section == "image" and b["width"] * b["height"] > 150_000:
            raw_ocr, ocr_conf = ocr_images([b["image"]])
            ocr = normalize_text(raw_ocr)
            if ocr and len(ocr.split()) >= 8:
                ochunks = chunk_by_tokens(ocr, max_tokens, overlap_tokens)
                print(f"[pipeline/docx] image ocr -> {len(ochunks)} chunks")
                for ch in ochunks:
                    yield new_chunk(doc_id=doc_id, source="docx", title=title,
                                    section=current_section, text=ch, caption="ocr",
                                    confidence=ocr_conf)
            num_blocks = 0
            continue

        # Regular paragraph — append to buffer and count
        if current_text:
            buf.append(current_text)
            num_blocks += 1

    # tail
    chunks = flush_buf()
    if chunks:
        print(f"[pipeline/docx] section '{previous_section}' -> {len(chunks)} chunks")
        for ch in chunks:
            yield new_chunk(doc_id=doc_id, source="docx", title=title, section=previous_section, text=ch)


def process_transcript_text(path: Path, max_tokens=512, overlap_tokens=80) -> Iterator[DocumentChunk]:
    doc_id = doc_id_from_path(path)
    title = guess_title_from_path(path)
    print(f"[pipeline/txt] start: {path}, doc_id={doc_id}")
    text = normalize_text(load_plain_text(str(path)))
    chunks = chunk_by_tokens(text, max_tokens, overlap_tokens)
    print(f"[pipeline/txt] -> {len(chunks)} chunks")
    for ch in chunks:
        yield new_chunk(doc_id=doc_id, source="transcript", title=title, text=ch)


# Final per-document chunk de-dup (safety net)
def dedup_chunks(chunks: Iterator[DocumentChunk]) -> Iterator[DocumentChunk]:
    seen: Set[str] = set()
    kept = 0
    for ch in chunks:
        sig = sha1_str(canonicalize_for_hash(ch.text))
        if sig in seen:
            continue
        seen.add(sig)
        kept += 1
        yield ch
    print(f"[pipeline/dedup] kept {kept} unique chunks after chunk-level de-dup")


def _fmt_hhmmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _probe_video_duration(path: Path) -> float:
    """
    Lightweight wrapper around ffprobe for duration in seconds.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to split video files but was not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed to read duration for {path}") from exc
    out = res.stdout.strip()
    try:
        return float(out)
    except ValueError as exc:
        raise RuntimeError(f"Unable to parse ffprobe duration output '{out}' for {path}") from exc


def _split_video_by_size(path: Path, duration: float) -> Tuple[List[Path], Optional[tempfile.TemporaryDirectory]]:
    """
    Split a video into <100MB segments using ffmpeg segmenter. Returns the
    list of segment paths and the temporary directory that owns them.
    """
    size_bytes = path.stat().st_size
    if size_bytes <= _VIDEO_SEGMENT_LIMIT_BYTES:
        # No split needed; return original path wrapped like a segment.
        return [path], None

    avg_bitrate = size_bytes / duration if duration > 0 else None
    if not avg_bitrate or math.isclose(avg_bitrate, 0.0):
        # Fallback to a conservative 5 minute segment when bitrate is unknown.
        target_segment_secs = 5 * 60
    else:
        target_segment_secs = max(5.0, (_VIDEO_SEGMENT_LIMIT_BYTES / avg_bitrate) * 0.98)

    tmpdir = tempfile.TemporaryDirectory(prefix=f"{path.stem}_segments_", dir=str(path.parent))
    tmp_path = Path(tmpdir.name)
    pattern = tmp_path / f"{path.stem}_part_%03d{path.suffix}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", f"{target_segment_secs:.3f}",
        "-reset_timestamps", "1",
        str(pattern)
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        tmpdir.cleanup()
        raise RuntimeError("ffmpeg is required to split video files but was not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        tmpdir.cleanup()
        raise RuntimeError(f"ffmpeg failed while splitting {path}") from exc

    segments = sorted(tmp_path.glob(f"{path.stem}_part_*{path.suffix}"))
    if not segments:
        tmpdir.cleanup()
        raise RuntimeError(f"ffmpeg did not produce segments for {path}")
    return segments, tmpdir


def process_video_mp4(path: Path, max_tokens=512, overlap_tokens=80) -> Iterator[DocumentChunk]:
    """
    MP4 -> Whisper segments -> blocks (~60s) -> normalize -> chunk
    """
    doc_id = doc_id_from_path(path)
    title  = guess_title_from_path(path)
    print(f"[pipeline/mp4] start (whisper): {path}, doc_id={doc_id}")

    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"[pipeline/mp4] file size: {size_mb:.2f} MB")

    try:
        duration = _probe_video_duration(path)
    except RuntimeError as err:
        raise RuntimeError(f"Failed to inspect video duration for {path}") from err

    segments, tmp_handle = _split_video_by_size(path, duration)
    cleanup_tmp = tmp_handle.cleanup if tmp_handle else None
    if tmp_handle:
        print(f"[pipeline/mp4] split into {len(segments)} segments (target <= {_VIDEO_SEGMENT_LIMIT_MB} MB each)")

    total_chunks = 0
    detected_language = None
    offset = 0.0

    try:
        for idx, segment_path in enumerate(segments, start=1):
            seg_bytes = segment_path.stat().st_size
            seg_mb = seg_bytes / (1024 * 1024)
            seg_duration = _probe_video_duration(segment_path)
            if segment_path is path:
                print(f"[pipeline/mp4] segment {idx}: original file ({seg_mb:.2f} MB, {seg_duration:.1f}s)")
            else:
                print(f"[pipeline/mp4] segment {idx}: {segment_path.name} ({seg_mb:.2f} MB, {seg_duration:.1f}s)")
                if seg_bytes > _VIDEO_SEGMENT_LIMIT_BYTES:
                    print(f"[pipeline/mp4] warning: segment {idx} exceeds {_VIDEO_SEGMENT_LIMIT_MB}MB ({seg_mb:.2f}MB)")

            asr = transcribe_whisper(str(segment_path))
            if not detected_language:
                detected_language = asr.get("language")

            blocks = segments_to_blocks(asr["segments"], max_secs=60.0, max_chars=1200, gap_break=1.5)
            for b in blocks:
                text = normalize_text(b["text"])
                if len(text.split()) < 8:
                    continue
                block_start = b["start"] + offset
                block_end = b["end"] + offset
                timecode = f"{_fmt_hhmmss(block_start)}-{_fmt_hhmmss(block_end)}"
                chunks = chunk_by_tokens(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
                print(f"[pipeline/mp4] block {timecode} -> {len(chunks)} chunks")
                for ch in chunks:
                    total_chunks += 1
                    yield new_chunk(doc_id=doc_id, source="transcript", title=title,
                                    timecode=timecode, text=ch)

            offset += seg_duration
    finally:
        if cleanup_tmp:
            cleanup_tmp()

    print(f"[pipeline/mp4] done: {total_chunks} chunks emitted (lang={detected_language})")

def process_file(path: str, max_tokens=512, overlap_tokens=80) -> Iterator[DocumentChunk]:
    p = Path(path)
    ext = p.suffix.lower()
    print(f"[pipeline/router] {path} (ext={ext})")
    if ext == ".pdf":
        gen = process_pdf(p, max_tokens, overlap_tokens)
    elif ext == ".pptx":
        gen = process_pptx(p, max_tokens, overlap_tokens)
    elif ext == ".docx":
        gen = process_docx(p, max_tokens, overlap_tokens)
    elif ext in {".txt", ".md"}:
        gen = process_transcript_text(p, max_tokens, overlap_tokens)
    elif ext == ".mp4":
        # transcripts typically don’t need de-dup; emit directly
        yield from process_video_mp4(p, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        return
    else:
        raise ValueError(f"Unsupported extension: {ext}. Supported: {SUPPORTED}")
    # for non-video docs, keep the final chunk-level de-dup safety net
    yield from dedup_chunks(gen)
