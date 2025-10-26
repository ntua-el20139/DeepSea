from typing import Iterator, Tuple, Optional, List, Dict, Any
from pathlib import Path
import io
from PIL import Image
import pdfplumber
from pypdf import PdfReader
from pptx import Presentation
from docx import Document as Docx

from ingest.docling_ocr import ocr_image

print("[loaders] module loaded")


def _pptx_table_to_markdown(shape) -> Optional[str]:
    """
    Return a markdown representation of the table contained in ``shape`` if it
    has one, otherwise ``None``.
    """
    try:
        table = shape.table
    except (AttributeError, ValueError):
        return None

    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(cells)

    if not rows:
        return None

    header = "| " + " | ".join(rows[0]) + " |"
    if len(rows) == 1:
        return header

    separator = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator, *body_lines]) if body_lines else header


def _rows_to_markdown(rows: List[List[str]]) -> Optional[str]:
    """
    Convert a 2D list of strings into a simple GitHub-style markdown table.
    Pads ragged rows to the max column width for consistent formatting.
    """
    if not rows:
        return None
    # Normalize and ensure all rows have the same length
    normalized: List[List[str]] = []
    max_cols = max(len(row) for row in rows)
    if max_cols == 0:
        return None
    for row in rows:
        cleaned = [(cell or "").strip() for cell in row]
        if len(cleaned) < max_cols:
            cleaned.extend([""] * (max_cols - len(cleaned)))
        normalized.append(cleaned)
    if not any(any(cell for cell in row) for row in normalized):
        return None
    header = "| " + " | ".join(normalized[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(max_cols)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in normalized[1:]]
    return "\n".join([header, separator, *body_lines]) if body_lines else header


def _pptx_image(shape) -> Optional[Image.Image]:
    """
    Return a PIL ``Image`` if ``shape`` embeds a picture, otherwise ``None``.
    """
    try:
        blob = shape.image.blob
    except (AttributeError, ValueError):
        return None

    try:
        return Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception:
        return None

def _docx_table_to_md(tbl) -> str:
    rows = [[cell.text.strip() for cell in row.cells] for row in tbl.rows]
    if not rows:
        return ""
    head = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join([head, sep, body]) if len(rows) > 1 else head

def load_pdf_text(path: str) -> Iterator[Tuple[int, str]]:
    print(f"[loaders] load_pdf_text: {path}")
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        yield (i + 1), (page.extract_text() or "")


def load_pdf_tables(path: str) -> Dict[int, List[str]]:
    """
    Extract tables from a PDF, returning markdown strings keyed by 1-based page number.
    """
    print(f"[loaders] load_pdf_tables: {path}")
    tables_by_page: Dict[int, List[str]] = {}
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                md_tables: List[str] = []
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                for tbl in tables:
                    if not tbl:
                        continue
                    # Ensure rows are consistent lists
                    rows = [list(row) for row in tbl if row]
                    md = _rows_to_markdown(rows) # type: ignore
                    if md:
                        md_tables.append(md)
                if md_tables:
                    tables_by_page[i + 1] = md_tables
    except Exception as e:
        print(f"[loaders] pdf table extraction failed ({e}); skipping")
    return tables_by_page

def _ocr_text_and_confidence(image: Image.Image) -> Tuple[str, Optional[float]]:
    """
    Return OCR text and an average confidence score for the provided image.
    Confidence is expressed on a 0–100 scale to match historical behaviour.
    """
    return ocr_image(image)


def load_pdf_images_ocr(path: str) -> Iterator[Tuple[int, str, Optional[float]]]:
    print(f"[loaders] load_pdf_images_ocr: {path}")
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                im = page.to_image(resolution=220).original
                txt, conf = _ocr_text_and_confidence(im)
                if txt.strip() and conf is not None and conf > 95.0:
                    yield (i + 1), txt, conf
    except Exception as e:
        print(f"[loaders] pdf OCR failed ({e}); skipping")
        return

def load_pptx(path: str) -> Iterator[Tuple[int, str, List[Image.Image], List[str]]]:
    """
    Yield (slide_number, concatenated_text, images[], tables_md[]) for each slide.
    """
    print(f"[loaders] load_pptx: {path}")
    prs = Presentation(path)
    for i, slide in enumerate(prs.slides):
        texts, images, tables_md = [], [], []
        for shape in slide.shapes:
            if hasattr(shape, "text") and getattr(shape, "has_text_frame", False):
                texts.append(shape.text) # type: ignore
            table_md = _pptx_table_to_markdown(shape)
            if table_md:
                tables_md.append(table_md)
            image = _pptx_image(shape)
            if image:
                images.append(image)
        yield (i + 1), "\n".join(texts), images, tables_md

def ocr_images(images: List[Image.Image]) -> Tuple[str, Optional[float]]:
    texts: List[str] = []
    confs: List[float] = []
    for im in images:
        txt, conf = _ocr_text_and_confidence(im)
        if txt.strip() and conf is not None and conf > 95.0:
            texts.append(txt)
            confs.append(conf)
    joined = "\n".join(texts).strip()
    avg_conf = (sum(confs) / len(confs)) if confs else None
    print(f"[loaders] ocr_images: {len(images)} images => {len(joined.split())} words (conf={avg_conf})")
    return joined, avg_conf

def load_plain_text(path: str) -> str:
    print(f"[loaders] load_plain_text: {path}")
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def load_docx_blocks(path: str) -> Iterator[Dict[str, Any]]:
    print(f"[loaders_docx] parsing: {path}")
    d = Docx(path)
    previous = {"status": None, "size": 12, "previous_empty_lines": 0}
    for p in d.paragraphs:
        size = int(p.runs[0].font.size.pt) if p.runs and p.runs[0].font.size else 12
        if p.style and p.style.name and p.style.name.lower().startswith("heading"):     # heading
            try:
                lvl = int(p.style.name.lower().replace("heading", "") or 1)
            except Exception:
                lvl = 1
            if p.text.strip():
                yield {"type": "heading", "level": lvl, "text": p.text}
                previous["status"] = "heading"
                previous["previous_empty_lines"] = 0
                previous["size"] = size

        elif p.text.strip():
            if size > previous["size"] and len(p.runs) == 1:       # bigger font
                if previous["status"] != "bigger_font" and p.text.strip():
                    yield {"type": "bigger_font", "text": p.text}
                previous["status"] = "bigger_font"
                
            elif p.runs[0].font.underline and len(p.runs) == 1:      # underline
                if previous["status"] != "underline" and p.text.strip():
                    yield {"type": "underline", "text": p.text}
                previous["status"] = "underline"
                
            elif p.runs[0].font.bold and len(p.runs) == 1:           # bold
                if previous["status"] != "bold" and p.text.strip():
                    yield {"type": "bold", "text": p.text}
                previous["status"] = "bold"
                
            elif p.runs[0].font.italic and len(p.runs) == 1:         # italic
                if previous["status"] != "italic" and p.text.strip():
                    yield {"type": "italic", "text": p.text}
                previous["status"] = "italic"
                
            elif previous["status"] == "empty" and previous["previous_empty_lines"] >= 2:     # paragraph break
                yield {"type": "paragraph_break", "text": p.text}
                previous["status"] = "paragraph_break"
                
            else:   # normal paragraph
                yield {"type": "paragraph", "text": p.text}
                previous["status"] = "paragraph"
            previous["previous_empty_lines"] = 0
            previous["size"] = size
            
        else:   # empty line
            previous["previous_empty_lines"] += 1
            previous["status"] = "empty"

    for t in d.tables:
        md = _docx_table_to_md(t)
        if md.strip():
            yield {"type": "table", "as_markdown": md}

    for shp in d.inline_shapes:
        try:
            rel_id = shp._inline.graphic.graphicData.pic.blipFill.blip.embed
            img_part = d.part.related_parts[rel_id]
            im = Image.open(io.BytesIO(img_part.blob)).convert("RGB")
            yield {"type": "image", "image": im, "width": im.width, "height": im.height}
        except Exception:
            continue
