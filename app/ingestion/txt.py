from pathlib import Path
from typing import Dict, Any
from app.ingestion.cleaner import clean_text

def parse_txt(file_path: Path) -> Dict[str, Any]:
    """
    Parses plain text / Markdown / JSON / CSV files.
    """
    raw_content = file_path.read_text(encoding='utf-8', errors='ignore')
    cleaned = clean_text(raw_content)
    
    return {
        "filename": file_path.name,
        "total_pages": 1,
        "raw_text": cleaned,
        "pages": [{"page": 1, "text": cleaned}]
    }
