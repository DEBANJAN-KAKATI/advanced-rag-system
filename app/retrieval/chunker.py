import bisect
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.core.config import settings

# Strategies selectable via settings.CHUNK_STRATEGY or the `strategy` argument.
CHUNK_STRATEGIES = ("structure", "fixed", "sentence", "semantic")

_HEADING = re.compile(r'^#+\s*(.+?)\s*$', re.MULTILINE)
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[`*])')


def chunk_document(
    doc_data: Dict[str, Any],
    strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    semantic_percentile: float = 90.0,
) -> List[Dict[str, Any]]:
    """
    Splits a parsed document into chunks.

    chunk_size / chunk_overlap are in (approximate) tokens, converted to
    characters at ~4 chars per token. Strategies:
      - structure: paragraph-packing with section-heading tracking (default)
      - fixed:     fixed-size character windows snapped to word boundaries
      - sentence:  sentence-packing with whole-sentence overlap
      - semantic:  breaks where adjacent-sentence embedding similarity drops
    """
    strategy = strategy or settings.CHUNK_STRATEGY
    size_chars = (chunk_size if chunk_size is not None else settings.CHUNK_SIZE) * 4
    overlap_chars = (chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP) * 4

    if strategy == "structure":
        return _chunk_structure(doc_data, size_chars, overlap_chars)
    if strategy == "fixed":
        return _chunk_fixed(doc_data, size_chars, overlap_chars)
    if strategy == "sentence":
        return _chunk_sentence(doc_data, size_chars, overlap_chars)
    if strategy == "semantic":
        return _chunk_semantic(doc_data, size_chars, embed_fn, semantic_percentile)
    raise ValueError(f"Unknown chunk strategy {strategy!r}; expected one of {CHUNK_STRATEGIES}")


def _make_chunk(doc_data: Dict[str, Any], index: int, page: int, section: str, text: str) -> Dict[str, Any]:
    return {
        "chunk_id": f"{doc_data['doc_id']}_c{index}",
        "doc_id": doc_data["doc_id"],
        "filename": doc_data["filename"],
        "page": page,
        "section": section,
        "text": text.strip(),
        "char_count": len(text),
    }


def _chunk_structure(doc_data: Dict[str, Any], target_chunk_chars: int, overlap_chars: int) -> List[Dict[str, Any]]:
    """
    Intelligent chunking strategy combining structure awareness (pages & section headings),
    paragraph preservation, and sliding window token/character limits with overlap.
    """
    chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for page_item in doc_data.get("pages", []):
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
                chunks.append(_make_chunk(doc_data, chunk_counter, page_num, current_section, current_chunk_text))
                chunk_counter += 1

                # Keep overlap from the end of current chunk
                overlap_text = current_chunk_text[-overlap_chars:] if overlap_chars and len(current_chunk_text) > overlap_chars else ""
                current_chunk_text = overlap_text + "\n" + para
            else:
                current_chunk_text += ("\n\n" if current_chunk_text else "") + para

        if current_chunk_text.strip():
            chunks.append(_make_chunk(doc_data, chunk_counter, page_num, current_section, current_chunk_text))
            chunk_counter += 1

    return chunks


class _SectionIndex:
    """Maps a character offset in a page to the nearest preceding heading."""

    def __init__(self, text: str):
        self.offsets = []
        self.titles = []
        for m in _HEADING.finditer(text):
            self.offsets.append(m.start())
            self.titles.append(m.group(1).strip('#').strip())

    def at(self, offset: int) -> str:
        i = bisect.bisect_right(self.offsets, offset) - 1
        return self.titles[i] if i >= 0 else "General"


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
                space = text.rfind(" ", start + step // 2, end)
                newline = text.rfind("\n", start + step // 2, end)
                cut = max(space, newline)
                if cut > start:
                    end = cut
            piece = text[start:end]
            if piece.strip():
                chunks.append(_make_chunk(doc_data, len(chunks), page_item.get("page", 1), sections.at(start), piece))
            if end >= len(text):
                break
            next_start = max(end - overlap_chars, start + 1)
            # Move forward to a word boundary for the overlap start too.
            while next_start < end and not text[next_start - 1].isspace():
                next_start += 1
            start = next_start
    return chunks


def split_sentences(text: str) -> List[tuple]:
    """Returns (offset, sentence) pairs. Paragraph breaks and headings always split."""
    units = []
    for para in re.finditer(r'(?:[^\n]|\n(?!\s*\n))+', text):
        para_text = para.group(0)
        if not para_text.strip():
            continue
        pos = 0
        for piece in _SENTENCE_END.split(para_text):
            idx = para_text.find(piece, pos)
            pos = idx + len(piece)
            if piece.strip():
                units.append((para.start() + idx, piece.strip()))
    return units


def _chunk_sentence(doc_data: Dict[str, Any], size_chars: int, overlap_chars: int) -> List[Dict[str, Any]]:
    """Packs whole sentences up to size_chars; overlap is whole trailing sentences."""
    chunks: List[Dict[str, Any]] = []
    for page_item in doc_data.get("pages", []):
        text = page_item.get("text", "")
        if not text.strip():
            continue
        sections = _SectionIndex(text)
        page = page_item.get("page", 1)
        current: List[tuple] = []
        current_len = 0
        for unit in split_sentences(text):
            if current and current_len + len(unit[1]) > size_chars:
                chunks.append(_make_chunk(doc_data, len(chunks), page, sections.at(current[0][0]),
                                          " ".join(s for _, s in current)))
                carry: List[tuple] = []
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
    Embeds every sentence and starts a new chunk where the cosine distance
    between neighbouring sentences is in the top (100 - percentile)% for the
    document, or where the chunk would exceed max_chars. Costs one embedding
    per sentence at index time.
    """
    if embed_fn is None:
        from app.retrieval.embedder import embedder
        embed_fn = embedder.embed_texts

    chunks: List[Dict[str, Any]] = []
    for page_item in doc_data.get("pages", []):
        text = page_item.get("text", "")
        if not text.strip():
            continue
        sections = _SectionIndex(text)
        page = page_item.get("page", 1)
        units = split_sentences(text)
        if not units:
            continue
        vectors = np.asarray(embed_fn([s for _, s in units]), dtype=np.float32)
        sims = np.sum(vectors[:-1] * vectors[1:], axis=1) if len(units) > 1 else np.array([])
        distances = 1.0 - sims
        threshold = float(np.percentile(distances, percentile)) if len(distances) else 1.0

        current: List[tuple] = []
        current_len = 0
        for i, unit in enumerate(units):
            breakpoint = i > 0 and distances[i - 1] > threshold
            if current and (breakpoint or current_len + len(unit[1]) > max_chars):
                chunks.append(_make_chunk(doc_data, len(chunks), page, sections.at(current[0][0]),
                                          " ".join(s for _, s in current)))
                current, current_len = [], 0
            current.append(unit)
            current_len += len(unit[1]) + 1
        if current:
            chunks.append(_make_chunk(doc_data, len(chunks), page, sections.at(current[0][0]),
                                      " ".join(s for _, s in current)))
    return chunks
