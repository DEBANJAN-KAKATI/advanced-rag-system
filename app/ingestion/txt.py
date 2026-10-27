import csv
import io
import json
from pathlib import Path
from typing import Dict, Any
from app.ingestion.cleaner import clean_text


def read_text_file(file_path: Path) -> str:
    """Reads a text file, trying common encodings before falling back to lossy UTF-8."""
    raw = file_path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            text = raw.decode(encoding)
            if encoding == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue  # only trust utf-16 when a BOM is present
            return text
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _single_page(file_path: Path, text: str) -> Dict[str, Any]:
    return {
        "filename": file_path.name,
        "total_pages": 1,
        "pages": [{"page": 1, "text": text}],
    }


def parse_txt(file_path: Path) -> Dict[str, Any]:
    """Parses plain text, Markdown, and source files."""
    return _single_page(file_path, clean_text(read_text_file(file_path)))


def parse_csv(file_path: Path) -> Dict[str, Any]:
    """
    Converts each CSV row into a self-describing "column: value" line so that
    every chunk keeps the meaning of its values even without the header row.
    """
    content = read_text_file(file_path)
    try:
        dialect = csv.Sniffer().sniff(content[:4096])
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(content), dialect))
    if not rows:
        return _single_page(file_path, "")

    header = [h.strip() or f"column_{i + 1}" for i, h in enumerate(rows[0])]
    lines = [f"Columns: {', '.join(header)}", ""]
    for row_num, row in enumerate(rows[1:], start=1):
        pairs = [f"{header[i] if i < len(header) else f'column_{i + 1}'}: {v.strip()}"
                 for i, v in enumerate(row) if v.strip()]
        if pairs:
            lines.append(f"Row {row_num}: " + "; ".join(pairs))
    return _single_page(file_path, clean_text("\n".join(lines)))


def parse_json(file_path: Path) -> Dict[str, Any]:
    """Pretty-prints JSON so nested keys stay readable; falls back to raw text if invalid."""
    content = read_text_file(file_path)
    try:
        content = json.dumps(json.loads(content), indent=2, ensure_ascii=False)
    except ValueError:
        pass
    return _single_page(file_path, clean_text(content))
