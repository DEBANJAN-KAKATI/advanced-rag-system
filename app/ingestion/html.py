from pathlib import Path
from typing import Dict, Any
from bs4 import BeautifulSoup
from app.ingestion.cleaner import clean_text
from app.ingestion.txt import read_text_file


def parse_html(file_path: Path) -> Dict[str, Any]:
    """
    Parses HTML into Markdown-like text: boilerplate tags are removed, headings
    become `#` lines (so the chunker can track sections) and tables become rows.
    """
    soup = BeautifulSoup(read_text_file(file_path), "html.parser")

    for element in soup(["script", "style", "nav", "header", "footer", "noscript", "svg", "form", "aside"]):
        element.decompose()

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(" | ".join(cells))
        table.replace_with("\n\n" + "\n".join(rows) + "\n\n")

    for level in range(1, 7):
        for heading in soup.find_all(f"h{level}"):
            text = heading.get_text(" ", strip=True)
            heading.replace_with(f"\n\n{'#' * level} {text}\n\n" if text else "")

    for li in soup.find_all("li"):
        li.insert_before("\n- ")

    for block in soup.find_all(["p", "div", "section", "article", "blockquote", "pre", "br"]):
        block.insert_after("\n\n" if block.name != "br" else "\n")

    title = soup.title.get_text(strip=True) if soup.title else ""
    if soup.title:
        soup.title.decompose()

    text = soup.get_text()
    if title:
        text = f"# {title}\n\n{text}"
    cleaned = clean_text(text)

    return {
        "filename": file_path.name,
        "total_pages": 1,
        "pages": [{"page": 1, "text": cleaned}],
    }
