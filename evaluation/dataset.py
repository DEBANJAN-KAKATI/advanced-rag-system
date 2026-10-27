"""
Golden dataset loading and chunking-independent relevance labelling.

Relevance is annotated as *evidence spans* (verbatim text from a source
document), never as chunk IDs. Chunk IDs change every time the chunker changes,
so labelling by span is what makes it possible to compare chunking strategies
against the same ground truth: a retrieved chunk counts as relevant to a span if
it contains that span (or most of it, when a chunk boundary cuts through it).
"""
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from app.ingestion.txt import parse_txt

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
DEFAULT_DATASET = EVAL_DIR / "datasets" / "pep_qa.jsonl"

# A chunk that contains at least this fraction of a span's word trigrams is
# treated as containing the span. Catches spans split across a chunk boundary
# while rejecting chunks that merely share vocabulary.
PARTIAL_MATCH_THRESHOLD = 0.5

_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _trigrams(text: str) -> set:
    words = _WORD.findall(text.lower())
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}


@dataclass
class Evidence:
    doc: str
    text: str

    @property
    def norm(self) -> str:
        return normalize(self.text)


@dataclass
class QAItem:
    id: str
    question: str
    type: str
    evidence: List[Evidence]
    evidence_mode: str = "all"  # "all": every span needed; "any": spans are interchangeable
    reference_answer: Optional[str] = None

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


@dataclass
class Corpus:
    """Documents as the app parses them, keyed by filename."""
    docs: Dict[str, dict] = field(default_factory=dict)


def load_dataset(path: Path = DEFAULT_DATASET) -> List[QAItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        items.append(QAItem(
            id=raw["id"],
            question=raw["question"],
            type=raw["type"],
            evidence=[Evidence(**e) for e in raw.get("evidence", [])],
            evidence_mode=raw.get("evidence_mode", "all"),
            reference_answer=raw.get("reference_answer"),
        ))
    return items


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> Corpus:
    """Parses every corpus file through the app's own ingestion code.

    doc_id is the file stem so IDs are stable across runs (the app derives
    doc_id from mtime, which would change on every checkout).
    """
    corpus = Corpus()
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        doc = parse_txt(path)
        doc["doc_id"] = path.stem
        doc["file_type"] = "txt"
        doc["size_bytes"] = path.stat().st_size
        corpus.docs[path.name] = doc
    return corpus


@lru_cache(maxsize=200_000)
def _span_coverage(span_norm: str, chunk_norm: str) -> float:
    if span_norm in chunk_norm:
        return 1.0
    span_grams = _trigrams(span_norm)
    if not span_grams:
        return 0.0
    chunk_grams = _trigrams(chunk_norm)
    return len(span_grams & chunk_grams) / len(span_grams)


def chunk_covers(chunk: dict, evidence: Evidence) -> bool:
    if chunk.get("filename") != evidence.doc:
        return False
    return _span_coverage(evidence.norm, normalize(chunk.get("text", ""))) >= PARTIAL_MATCH_THRESHOLD


def validate(items: List[QAItem], corpus: Corpus) -> List[str]:
    """Returns a list of problems; empty means every span is verbatim in its doc."""
    problems = []
    seen = set()
    for item in items:
        if item.id in seen:
            problems.append(f"{item.id}: duplicate id")
        seen.add(item.id)
        if item.answerable and item.evidence_mode not in ("all", "any"):
            problems.append(f"{item.id}: bad evidence_mode {item.evidence_mode!r}")
        for ev in item.evidence:
            doc = corpus.docs.get(ev.doc)
            if doc is None:
                problems.append(f"{item.id}: unknown doc {ev.doc}")
            elif ev.norm not in normalize(doc["raw_text"]):
                problems.append(f"{item.id}: span not found verbatim in {ev.doc}: {ev.text[:60]!r}")
    return problems
