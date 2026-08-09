import uuid
import hashlib
from pathlib import Path
from typing import Dict, Any
from app.utils.filetypes import get_file_type
from app.ingestion.pdf import parse_pdf
from app.ingestion.docx import parse_docx
from app.ingestion.html import parse_html
from app.ingestion.txt import parse_txt
from app.utils.logging import logger

def generate_doc_id(file_path: Path) -> str:
    """Generates a stable document ID from filename and size."""
    stat = file_path.stat()
    raw = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def load_document(file_path: Path) -> Dict[str, Any]:
    """
    Detects file type and loads document contents and metadata.
    """
    file_type = get_file_type(file_path.name)
    doc_id = generate_doc_id(file_path)
    
    logger.info(f"Loading document: {file_path.name} (type: {file_type}, doc_id: {doc_id})")
    
    if file_type == "pdf":
        result = parse_pdf(file_path)
    elif file_type == "docx":
        result = parse_docx(file_path)
    elif file_type == "html":
        result = parse_html(file_path)
    else:
        result = parse_txt(file_path)
        
    result["doc_id"] = doc_id
    result["file_type"] = file_type
    result["size_bytes"] = file_path.stat().st_size
    
    return result
