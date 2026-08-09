import re

def clean_text(text: str) -> str:
    """Clean and normalize extracted document text."""
    if not text:
        return ""
        
    # Replace non-breaking spaces and control characters
    text = text.replace('\xa0', ' ').replace('\r\n', '\n').replace('\r', '\n')
    
    # Strip invisible zero-width characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Remove repeated page header/footer line patterns like "Page 1 of 10" or "--- Page 1 ---"
    text = re.sub(r'(?i)^\s*(?:page|\d+)\s+(?:of\s+)?\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-=_]{3,}\s*page\s*\d+\s*[-=_]{3,}\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Fix multiple blank lines (keep max 2 newlines for paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive horizontal spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # Clean OCR glitch repetitions e.g. "aaaaa" or "....."
    text = re.sub(r'(\.\s*){5,}', '... ', text)
    text = re.sub(r'(,\s*){4,}', ', ', text)
    
    return text.strip()

def extract_sections(text: str) -> list[dict]:
    """Break text into logical sections based on headings."""
    lines = text.split('\n')
    sections = []
    current_heading = "General"
    current_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Heuristic for section heading: short uppercase/titleline or markdown heading
        if (stripped.startswith('#') or 
            (len(stripped) > 0 and len(stripped) < 80 and stripped.isupper() and len(stripped.split()) < 8)):
            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "text": "\n".join(current_lines).strip()
                })
                current_lines = []
            current_heading = stripped.lstrip('#').strip()
        else:
            current_lines.append(line)
            
    if current_lines:
        sections.append({
            "heading": current_heading,
            "text": "\n".join(current_lines).strip()
        })
        
    return [s for s in sections if s["text"]]
