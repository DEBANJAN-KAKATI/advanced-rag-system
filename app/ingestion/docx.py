from pathlib import Path
from typing import Dict, Any, List
import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from app.ingestion.cleaner import clean_text


def _iter_block_items(document):
    """Yields paragraphs and tables in document order (tables used to be appended at the end)."""
    if hasattr(document, "iter_inner_content"):
        yield from document.iter_inner_content()
        return
    body = document.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def _heading_level(paragraph: Paragraph) -> int:
    name = (paragraph.style.name if paragraph.style is not None else "") or ""
    if name == "Title":
        return 1
    if name.startswith("Heading"):
        digits = "".join(ch for ch in name if ch.isdigit())
        return max(1, min(int(digits), 6)) if digits else 2
    return 0


def _table_to_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells: List[str] = []
        for cell in row.cells:
            text = cell.text.strip().replace("\n", " ")
            if not cells or cells[-1] != text:  # merged cells repeat their text
                cells.append(text)
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_docx(file_path: Path) -> Dict[str, Any]:
    """Parses a DOCX document, keeping headings, list items and tables in reading order."""
    document = docx.Document(str(file_path))
    blocks: List[str] = []

    for item in _iter_block_items(document):
        if isinstance(item, Table):
            table_text = _table_to_text(item)
            if table_text:
                blocks.append(table_text)
            continue

        text = item.text.strip()
        if not text:
            continue
        level = _heading_level(item)
        style_name = (item.style.name if item.style is not None else "") or ""
        if level:
            blocks.append(f"{'#' * level} {text}")
        elif "List" in style_name:
            blocks.append(f"- {text}")
        else:
            blocks.append(text)

    cleaned = clean_text("\n\n".join(blocks))
    return {
        "filename": file_path.name,
        "total_pages": 1,
        "pages": [{"page": 1, "text": cleaned}],
    }
