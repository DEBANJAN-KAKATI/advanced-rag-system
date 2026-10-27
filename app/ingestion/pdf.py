from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from app.ingestion.cleaner import clean_text, fix_pdf_line_breaks
from app.ingestion.ocr import run_ocr_on_image_bytes
from app.utils.logging import logger

# Pages with less extracted text than this are treated as scanned and sent to OCR.
MIN_PAGE_TEXT_CHARS = 20


def parse_pdf(file_path: Path) -> Dict[str, Any]:
    """
    Parses a PDF page by page, with OCR fallback for pages whose digital text is
    empty or near-empty (scanned pages).
    """
    reader = PdfReader(str(file_path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as e:
            raise ValueError("This PDF is password-protected and cannot be read.") from e

    pages_data: List[Dict[str, Any]] = []
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Text extraction failed on page {page_num} of {file_path.name}: {e}")
            page_text = ""
        cleaned = clean_text(fix_pdf_line_breaks(page_text))

        if len(cleaned) < MIN_PAGE_TEXT_CHARS:
            try:
                images = list(page.images)
            except Exception:
                images = []
            if images:
                logger.info(f"Page {page_num} of {file_path.name} has little text; running OCR on {len(images)} image(s).")
                ocr_texts = [t for t in (run_ocr_on_image_bytes(img.data) for img in images) if t]
                if ocr_texts:
                    cleaned = clean_text("\n\n".join(ocr_texts))

        pages_data.append({"page": page_num, "text": cleaned})

    return {
        "filename": file_path.name,
        "total_pages": len(reader.pages),
        "pages": pages_data,
    }
