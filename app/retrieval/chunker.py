import re
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import settings

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"((?<=[.!?;:])[ \t]+(?=[\"'(\[A-Z0-9])|\s*\n\s*)")


def detect_heading(line: str) -> Optional[Tuple[int, str]]:
    """Returns (level, title) if the line looks like a heading, else None."""
    stripped = line.strip()
    md = _MD_HEADING_RE.match(stripped)
    if md:
        return len(md.group(1)), md.group(2).strip()
    # Short ALL-CAPS lines ("KEY RISK FACTORS") are common headings in PDFs.
    if (3 <= len(stripped) <= 80 and stripped.isupper() and len(stripped.split()) <= 10
            and re.search(r"[A-Z]{3}", stripped) and not stripped.endswith((".", ",", ";"))):
        return 2, stripped
    return None


def split_sentences(text: str) -> List[Tuple[str, str]]:
    """
    Splits text into (sentence, separator) pairs, where separator is the whitespace
    that preceded the sentence ("\n" for line breaks, " " otherwise).
    """
    parts = _SENTENCE_BOUNDARY_RE.split(text)
    result: List[Tuple[str, str]] = []
    sep = ""
    for i, part in enumerate(parts):
        if i % 2 == 1:
            sep = "\n" if "\n" in part else " "
        elif part.strip():
            result.append((part.strip(), sep if result else ""))
    return result


def split_paragraph(text: str, max_chars: int) -> List[Tuple[str, str]]:
    """
    Returns (piece, separator) pairs for a paragraph. Short paragraphs stay whole;
    long ones are split into sentences, and over-long sentences into word windows.
    """
    if len(text) <= max_chars:
        return [(text, "")]
    pieces: List[Tuple[str, str]] = []
    for sentence, sep in split_sentences(text):
        if len(sentence) <= max_chars:
            pieces.append((sentence, sep))
            continue
        window, window_sep = "", sep
        for word in sentence.split():
            if window and len(window) + 1 + len(word) > max_chars:
                pieces.append((window, window_sep))
                window, window_sep = word, " "
            else:
                window = f"{window} {word}" if window else word
        if window:
            pieces.append((window, window_sep))
    return pieces


def _tail_overlap(text: str, overlap_chars: int) -> str:
    """Whole sentences from the end of `text`, up to roughly overlap_chars."""
    if overlap_chars <= 0:
        return ""
    picked: List[str] = []
    total = 0
    for sentence, sep in reversed(split_sentences(text)):
        if total + len(sentence) > overlap_chars:
            break
        picked.append(sep + sentence)
        total += len(sentence) + 1
    return "".join(reversed(picked)).strip()


def _iter_units(pages: List[Dict[str, Any]]):
    """Yields ("heading", (level, title), page) and ("text", paragraph, page) units in order."""
    for page_item in pages:
        page_num = page_item.get("page", 1)
        page_text = page_item.get("text", "") or ""
        for block in re.split(r"\n\s*\n", page_text):
            lines: List[str] = []
            for line in block.split("\n"):
                heading = detect_heading(line)
                if heading:
                    if lines:
                        yield "text", "\n".join(lines).strip(), page_num
                        lines = []
                    yield "heading", heading, page_num
                elif line.strip():
                    lines.append(line.rstrip())
            if lines:
                yield "text", "\n".join(lines).strip(), page_num


def chunk_document(doc_data: Dict[str, Any], chunk_size: Optional[int] = None,
                   chunk_overlap: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Structure-aware chunking:
    - paragraphs are packed into chunks of up to `chunk_size` characters;
    - oversized paragraphs are split at sentence (then word) boundaries;
    - a new section heading starts a new chunk, and every chunk records its section path;
    - consecutive chunks share whole-sentence overlap; chunks may span pages (page..page_end).
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    overlap = min(overlap, chunk_size // 2)
    min_chunk = max(chunk_size // 4, 1)

    doc_id = doc_data["doc_id"]
    filename = doc_data["filename"]

    chunks: List[Dict[str, Any]] = []
    section_stack: List[Tuple[int, str]] = []
    buf: List[Tuple[str, int]] = []  # (text, page)
    buf_section = ""
    has_new_content = False

    def current_section() -> str:
        return " > ".join(title for _, title in section_stack[-2:])

    def flush(keep_overlap: bool):
        nonlocal buf, buf_section, has_new_content
        if buf and has_new_content:
            text = "\n\n".join(t for t, _ in buf).strip()
            chunks.append({
                "chunk_id": f"{doc_id}_c{len(chunks)}",
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": len(chunks),
                "page": buf[0][1],
                "page_end": buf[-1][1],
                "section": buf_section or "General",
                "text": text,
                "char_count": len(text),
            })
            tail = _tail_overlap(text, overlap) if keep_overlap else ""
            buf = [(tail, buf[-1][1])] if tail else []
        else:
            buf = [] if not keep_overlap else buf
        buf_section = current_section()
        has_new_content = False

    def buf_len() -> int:
        return sum(len(t) + 2 for t, _ in buf)

    for kind, value, page in _iter_units(doc_data.get("pages", [])):
        if kind == "heading":
            level, title = value
            if has_new_content and buf_len() >= min_chunk:
                flush(keep_overlap=False)
            elif not has_new_content:
                buf = []  # drop overlap carried from the previous section
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, title))
            if not has_new_content:
                buf_section = current_section()
            buf.append(("#" * level + " " + title, page))
            has_new_content = True
            continue

        # Pieces are budgeted so that overlap + piece never exceeds chunk_size.
        for piece, sep in split_paragraph(value, max(chunk_size - overlap, min_chunk)):
            if has_new_content and buf_len() + len(piece) > chunk_size:
                flush(keep_overlap=True)
                sep = ""
            if not buf:
                buf_section = current_section()
            if sep and buf:
                # Continuation of the same paragraph: keep it in the same block.
                buf[-1] = (buf[-1][0] + sep + piece, buf[-1][1])
            else:
                buf.append((piece, page))
            has_new_content = True

    flush(keep_overlap=False)
    return chunks


def chunk_search_text(chunk: Dict[str, Any]) -> str:
    """
    Text used for embedding and keyword search: the chunk prefixed with its document
    and section, so passages like "It increased 24%" still match "Q3 revenue" queries.
    """
    header = chunk.get("filename", "")
    section = chunk.get("section")
    if section and section != "General":
        header = f"{header} | {section}"
    return f"{header}\n{chunk.get('text', '')}"
