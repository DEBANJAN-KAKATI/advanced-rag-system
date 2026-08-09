import uuid
import re
from typing import List, Dict, Any
from app.core.config import settings

def chunk_document(doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Intelligent chunking strategy combining structure awareness (pages & section headings),
    paragraph preservation, and sliding window token/character limits with overlap.
    """
    doc_id = doc_data["doc_id"]
    filename = doc_data["filename"]
    pages = doc_data.get("pages", [])
    
    target_chunk_chars = settings.CHUNK_SIZE * 4  # Rough token to char ratio ~1:4
    overlap_chars = settings.CHUNK_OVERLAP * 4
    
    chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for page_item in pages:
        page_num = page_item.get("page", 1)
        page_text = page_item.get("text", "")
        if not page_text:
            continue
            
        # Split page text by double line breaks (paragraphs)
        paragraphs = re.split(r'\n{2,}', page_text)
        
        current_chunk_text = ""
        current_section = "General"
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Check for section header
            if para.startswith('#') or (len(para) < 80 and para.isupper()):
                current_section = para.strip('#').strip()
                
            # If adding paragraph exceeds chunk limit, flush current chunk
            if len(current_chunk_text) + len(para) > target_chunk_chars and len(current_chunk_text) > 50:
                chunk_id = f"{doc_id}_c{chunk_counter}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page_num,
                    "section": current_section,
                    "text": current_chunk_text.strip(),
                    "char_count": len(current_chunk_text)
                })
                chunk_counter += 1
                
                # Keep overlap from the end of current chunk
                overlap_text = current_chunk_text[-overlap_chars:] if len(current_chunk_text) > overlap_chars else ""
                current_chunk_text = overlap_text + "\n" + para
            else:
                current_chunk_text += ("\n\n" if current_chunk_text else "") + para
                
        if current_chunk_text.strip():
            chunk_id = f"{doc_id}_c{chunk_counter}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "filename": filename,
                "page": page_num,
                "section": current_section,
                "text": current_chunk_text.strip(),
                "char_count": len(current_chunk_text)
            })
            chunk_counter += 1

    return chunks
