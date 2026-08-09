from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from app.ingestion.cleaner import clean_text
from app.ingestion.ocr import run_ocr_on_image_bytes
from app.utils.logging import logger

def parse_pdf(file_path: Path) -> Dict[str, Any]:
    """
    Parses a PDF document, page by page.
    Extracts text and page numbers, with OCR fallback if digital text is poor or empty.
    Returns metadata and structured pages.
    """
    reader = PdfReader(str(file_path))
    total_pages = len(reader.pages)
    pages_data: List[Dict[str, Any]] = []
    full_text_list = []

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text() or ""
        cleaned = clean_text(page_text)

        # OCR Fallback if page text is very short/empty and images exist
        if len(cleaned) < 20 and len(page.images) > 0:
            logger.info(f"Page {page_num} of {file_path.name} has low text ({len(cleaned)} chars). Attempting OCR on page images.")
            ocr_texts = []
            for img in page.images:
                try:
                    ocr_res = run_ocr_on_image_bytes(img.data)
                    if ocr_res:
                        ocr_texts.append(ocr_res)
                except Exception as e:
                    logger.warning(f"Failed to extract image from page {page_num}: {e}")
            if ocr_texts:
                cleaned = clean_text("\n".join(ocr_texts))

        pages_data.append({
            "page": page_num,
            "text": cleaned
        })
        if cleaned:
            full_text_list.append(f"--- Page {page_num} ---\n{cleaned}")

    full_text = "\n\n".join(full_text_list)
    
    return {
        "filename": file_path.name,
        "total_pages": total_pages,
        "raw_text": full_text,
        "pages": pages_data
    }
