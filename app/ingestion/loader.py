import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from app.utils.filetypes import get_file_type
from app.ingestion.pdf import parse_pdf
from app.ingestion.docx import parse_docx
from app.ingestion.html import parse_html
from app.ingestion.txt import parse_txt, parse_csv, parse_json
from app.utils.logging import logger

PARSERS = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "html": parse_html,
    "csv": parse_csv,
    "json": parse_json,
    "txt": parse_txt,
}


class EmptyDocumentError(ValueError):
    """Raised when no readable text could be extracted from a document."""


def compute_doc_id(file_path: Path) -> str:
    """
    Content-addressed document ID: the same bytes always map to the same ID,
    so re-uploading an identical file is detected instead of duplicated.
    """
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def load_document(file_path: Path, display_name: Optional[str] = None, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Detects the file type, parses the document into pages and attaches metadata.
    `display_name` is the user's original filename when the file is stored under another name.
    """
    filename = display_name or file_path.name
    file_type = get_file_type(filename)
    doc_id = doc_id or compute_doc_id(file_path)

    logger.info(f"Loading document: {filename} (type: {file_type}, doc_id: {doc_id})")
    result = PARSERS.get(file_type, parse_txt)(file_path)

    if not any(p.get("text", "").strip() for p in result.get("pages", [])):
        hint = (" It may be a scanned PDF; install Tesseract OCR and `pytesseract` to read scanned pages."
                if file_type == "pdf" else "")
        raise EmptyDocumentError(f"No readable text could be extracted from '{filename}'.{hint}")

    result["filename"] = filename
    result["doc_id"] = doc_id
    result["file_type"] = file_type
    result["size_bytes"] = file_path.stat().st_size
    return result
