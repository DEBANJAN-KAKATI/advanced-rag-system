from pathlib import Path
from typing import Dict, Any
from bs4 import BeautifulSoup
from app.ingestion.cleaner import clean_text

def parse_html(file_path: Path) -> Dict[str, Any]:
    """
    Parses HTML content, stripping script, style, header, footer, nav tags
    and keeping visible structured text.
    """
    content = file_path.read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(content, 'html.parser')
    
    # Remove script, style, nav, header, footer elements
    for element in soup(["script", "style", "nav", "header", "footer", "noscript", "svg"]):
        element.decompose()
        
    text = soup.get_text(separator='\n')
    cleaned = clean_text(text)
    
    return {
        "filename": file_path.name,
        "total_pages": 1,
        "raw_text": cleaned,
        "pages": [{"page": 1, "text": cleaned}]
    }
