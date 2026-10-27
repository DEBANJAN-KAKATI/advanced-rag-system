import re
from typing import List, Dict, Any, Tuple, Set
from app.core.schemas import Citation
from app.generation.prompts import NOT_FOUND_SENTENCE

_CITATION_GROUP_RE = re.compile(r"\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")
_NOT_FOUND_MARKERS = (
    NOT_FOUND_SENTENCE.lower().rstrip("."),
    "couldn't find this in the uploaded documents",
    "could not find this in the uploaded documents",
    "insufficient evidence",
    "do not contain sufficient",
)


def normalize_citation_markers(answer: str, source_count: int) -> str:
    """
    Rewrites grouped markers like "[1, 3]" as "[1][3]" so each one can be linked,
    and drops numbers that do not correspond to a provided source.
    """
    def replace(match):
        numbers = [n.strip() for n in re.split(r"[,;]", match.group(2))]
        markers = "".join(f"[{n}]" for n in numbers if 1 <= int(n) <= source_count)
        return match.group(1) + markers if markers else ""

    return re.sub(r"([ \t]?)" + _CITATION_GROUP_RE.pattern, replace, answer)


def extract_cited_numbers(answer: str) -> Set[int]:
    numbers: Set[int] = set()
    for group in _CITATION_GROUP_RE.findall(answer):
        numbers.update(int(n) for n in re.split(r"\s*[,;]\s*", group) if n.strip().isdigit())
    return numbers


def is_not_found_answer(answer: str) -> bool:
    head = answer.strip().lower()[:300].replace("’", "'")
    return any(marker in head for marker in _NOT_FOUND_MARKERS)


def build_citations(scored_chunks: List[Tuple[Dict[str, Any], float]]) -> List[Citation]:
    """Numbers passages 1..n in the order they were given to the model."""
    citations = []
    for idx, (chunk, score) in enumerate(scored_chunks):
        text = chunk.get("text", "")
        citations.append(Citation(
            id=f"source-{idx + 1}",
            number=idx + 1,
            chunk_id=chunk.get("chunk_id", ""),
            doc_id=chunk.get("doc_id", ""),
            filename=chunk.get("filename", "Unknown Document"),
            page=chunk.get("page", 1),
            page_end=chunk.get("page_end") or chunk.get("page", 1),
            section=chunk.get("section", "General"),
            snippet=text[:300] + ("..." if len(text) > 300 else ""),
            text=text,
            score=round(float(score), 4),
        ))
    return citations


def select_cited(citations: List[Citation], answer: str) -> List[Citation]:
    """Keeps only the sources the answer actually cites (all of them if it cites none)."""
    if is_not_found_answer(answer):
        return []
    cited = extract_cited_numbers(answer)
    selected = [c for c in citations if c.number in cited]
    return selected or citations


def evaluate_confidence(answer: str, scored_chunks: List[Tuple[Dict[str, Any], float]],
                        scorer: str, low_relevance: bool, mode: str) -> str:
    """
    Confidence from the reranker's relevance of the best passage, downgraded when the
    retrieval was weak or the answer itself reports missing evidence.
    """
    if not scored_chunks or is_not_found_answer(answer):
        return "Insufficient Evidence"
    if mode == "summary":
        return "High"
    if low_relevance:
        return "Low"
    top = scored_chunks[0][1]
    high, medium = (0.5, 0.1) if scorer == "cross-encoder" else (0.55, 0.35)
    if top >= high:
        return "High"
    if top >= medium:
        return "Medium"
    return "Low"
