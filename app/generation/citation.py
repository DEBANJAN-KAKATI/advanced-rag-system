import re
from typing import List, Dict, Any, Tuple
from app.core.schemas import Citation

def build_citations(retrieved_chunks: List[Tuple[Dict[str, Any], float]]) -> List[Citation]:
    """Converts retrieved candidate chunks into formal Citation objects."""
    citations = []
    for idx, (chunk, score) in enumerate(retrieved_chunks):
        cite_id = f"source-{idx + 1}"
        citations.append(Citation(
            id=cite_id,
            doc_id=chunk.get("doc_id", ""),
            filename=chunk.get("filename", "Unknown Document"),
            page=chunk.get("page", 1),
            section=chunk.get("section", "General"),
            snippet=chunk.get("text", "")[:350] + ("..." if len(chunk.get("text", "")) > 350 else ""),
            score=round(float(score), 4)
        ))
    return citations

def evaluate_confidence(answer_text: str, retrieved_chunks: List[Tuple[Dict[str, Any], float]]) -> str:
    """
    Evaluates confidence score of the generated answer based on retrieval scores
    and text presence of insufficient evidence phrases.
    """
    if not retrieved_chunks:
        return "Insufficient Evidence"
        
    lower_answer = answer_text.lower()
    if "insufficient evidence" in lower_answer or "do not contain sufficient" in lower_answer:
        return "Insufficient Evidence"

    top_score = retrieved_chunks[0][1] if retrieved_chunks else 0.0
    avg_score = sum(s for _, s in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0.0

    if top_score > 0.5 or avg_score > 0.35:
        return "High"
    elif top_score > 0.2 or avg_score > 0.15:
        return "Medium"
    else:
        return "Low"
