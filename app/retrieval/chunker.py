import bisect
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from app.core.config import settings

# Selectable via settings.CHUNK_STRATEGY or the `strategy` argument; compared in docs/EVALUATION_REPORT.md.
CHUNK_STRATEGIES = ("structure", "fixed", "sentence", "semantic")

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_HEADING_LINE_RE = re.compile(_MD_HEADING_RE.pattern, re.MULTILINE)
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


def chunk_document(
    doc_data: Dict[str, Any],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    strategy: Optional[str] = None,
    embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    semantic_percentile: float = 90.0,
) -> List[Dict[str, Any]]:
    """
    Splits a parsed document into chunks. Sizes are in characters (~4 per token).

    Strategies:
      - structure: paragraph packing with section tracking (default, see _chunk_structure)
      - fixed:     fixed-size character windows snapped to word boundaries
      - sentence:  sentence packing with whole-sentence overlap
      - semantic:  breaks where adjacent-sentence embedding similarity drops
                   (embeds every sentence, so indexing costs more)
    """
    strategy = strategy or settings.CHUNK_STRATEGY
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    if strategy == "structure":
        return _chunk_structure(doc_data, chunk_size, overlap)
    if strategy == "fixed":
        return _chunk_fixed(doc_data, chunk_size, overlap)
    if strategy == "sentence":
        return _chunk_sentence(doc_data, chunk_size, overlap)
    if strategy == "semantic":
        return _chunk_semantic(doc_data, chunk_size, embed_fn, semantic_percentile)
    raise ValueError(f"Unknown chunk strategy {strategy!r}; expected one of {CHUNK_STRATEGIES}")


def _chunk_structure(doc_data: Dict[str, Any], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    """
    Structure-aware chunking:
    - paragraphs are packed into chunks of up to `chunk_size` characters;
    - oversized paragraphs are split at sentence (then word) boundaries;
    - a new section heading starts a new chunk, and every chunk records its section path;
    - consecutive chunks share whole-sentence overlap; chunks may span pages (page..page_end).
    """
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


# ---------------------------------------------------------------------------- alternative strategies
def _make_chunk(doc_data: Dict[str, Any], index: int, page: int, section: str, text: str) -> Dict[str, Any]:
    text = text.strip()
    return {
        "chunk_id": f"{doc_data['doc_id']}_c{index}",
        "doc_id": doc_data["doc_id"],
        "filename": doc_data["filename"],
        "chunk_index": index,
        "page": page,
        "page_end": page,
        "section": section,
        "text": text,
        "char_count": len(text),
    }


class _SectionIndex:
    """Maps a character offset in a page to the nearest preceding Markdown heading."""

    def __init__(self, text: str):
        self.offsets, self.titles = [], []
        for m in _MD_HEADING_LINE_RE.finditer(text):
            self.offsets.append(m.start())
            self.titles.append(m.group(2).strip())

    def at(self, offset: int) -> str:
        i = bisect.bisect_right(self.offsets, offset) - 1
        return self.titles[i] if i >= 0 else "General"


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`*])")


def sentence_spans(text: str) -> List[Tuple[int, str]]:
    """Returns (offset, sentence) pairs. Paragraph breaks always split."""
    units = []
    for para in re.finditer(r"(?:[^\n]|\n(?!\s*\n))+", text):
        para_text = para.group(0)
        if not para_text.strip():
            continue
        pos = 0
        for piece in _SENTENCE_END_RE.split(para_text):
            idx = para_text.find(piece, pos)
            pos = idx + len(piece)
            if piece.strip():
                units.append((para.start() + idx, piece.strip()))
    return units


def _chunk_fixed(doc_data: Dict[str, Any], size_chars: int, overlap_chars: int) -> List[Dict[str, Any]]:
    """Fixed-size windows that ignore document structure (the naive baseline)."""
    chunks: List[Dict[str, Any]] = []
    step = max(1, size_chars - overlap_chars)
    for page_item in doc_data.get("pages", []):
        text = page_item.get("text", "")
        if not text.strip():
            continue
        sections = _SectionIndex(text)
        start = 0
        while start < len(text):
            end = min(len(text), start + size_chars)
            if end < len(text):
                # Snap to the last whitespace so words are not cut in half.
                cut = max(text.rfind(" ", start + step // 2, end), text.rfind("\n", start + step // 2, end))
                if cut > start:
                    end = cut
            piece = text[start:end]
            if piece.strip():
                chunks.append(_make_chunk(doc_data, len(chunks), page_item.get("page", 1), sections.at(start), piece))
            if end >= len(text):
                break
            next_start = max(end - overlap_chars, start + 1)
            while next_start < end and not text[next_start - 1].isspace():
                next_start += 1
            start = next_start
    return chunks


def _chunk_sentence(doc_data: Dict[str, Any], size_chars: int, overlap_chars: int) -> List[Dict[str, Any]]:
    """Packs whole sentences up to size_chars; overlap is whole trailing sentences."""
    chunks: List[Dict[str, Any]] = []
    for page_item in doc_data.get("pages", []):
        text = page_item.get("text", "")
        if not text.strip():
            continue
        sections = _SectionIndex(text)
        page = page_item.get("page", 1)
        current: List[Tuple[int, str]] = []
        current_len = 0
        for unit in sentence_spans(text):
            if current and current_len + len(unit[1]) > size_chars:
                chunks.append(_make_chunk(doc_data, len(chunks), page, sections.at(current[0][0]),
                                          " ".join(s for _, s in current)))
                carry: List[Tuple[int, str]] = []
                carry_len = 0
                for prev in reversed(current):
                    if carry_len + len(prev[1]) > overlap_chars:
                        break
                    carry.insert(0, prev)
                    carry_len += len(prev[1]) + 1
                current, current_len = carry, carry_len
            current.append(unit)
            current_len += len(unit[1]) + 1
        if current:
            chunks.append(_make_chunk(doc_data, len(chunks), page, sections.at(current[0][0]),
                                      " ".join(s for _, s in current)))
    return chunks


def _chunk_semantic(doc_data: Dict[str, Any], max_chars: int,
                    embed_fn: Optional[Callable[[List[str]], np.ndarray]],
                    percentile: float) -> List[Dict[str, Any]]:
    """
    Embeds every sentence and starts a new chunk where the cosine distance between
    neighbouring sentences is in the top (100 - percentile)% for the page, or where
    the chunk would exceed max_chars.
    """
    if embed_fn is None:
        from app.retrieval.vectorstore import vector_store
        if vector_store.embedder is None:
            raise ValueError("Semantic chunking needs an embedding model.")
        embed_fn = vector_store.embedder.embed_documents

    chunks: List[Dict[str, Any]] = []
    for page_item in doc_data.get("pages", []):
        text = page_item.get("text", "")
        if not text.strip():
            continue
        sections = _SectionIndex(text)
        page = page_item.get("page", 1)
        units = sentence_spans(text)
        if not units:
            continue
        vectors = np.asarray(embed_fn([s for _, s in units]), dtype=np.float32)
        sims = np.sum(vectors[:-1] * vectors[1:], axis=1) if len(units) > 1 else np.array([])
        distances = 1.0 - sims
        threshold = float(np.percentile(distances, percentile)) if len(distances) else 1.0

        current: List[Tuple[int, str]] = []
        current_len = 0
        for i, unit in enumerate(units):
            is_break = i > 0 and distances[i - 1] > threshold
            if current and (is_break or current_len + len(unit[1]) > max_chars):
                chunks.append(_make_chunk(doc_data, len(chunks), page, sections.at(current[0][0]),
                                          " ".join(s for _, s in current)))
                current, current_len = [], 0
            current.append(unit)
            current_len += len(unit[1]) + 1
        if current:
            chunks.append(_make_chunk(doc_data, len(chunks), page, sections.at(current[0][0]),
                                      " ".join(s for _, s in current)))
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
