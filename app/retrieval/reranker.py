import threading
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from app.core.config import settings
from app.retrieval.keyword import term_coverage
from app.utils.logging import logger

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


def reciprocal_rank_fusion(ranked_lists: List[List[Tuple[Dict[str, Any], float]]], k: int = 60) -> List[Tuple[Dict[str, Any], float]]:
    """Merges several ranked lists: score(chunk) = sum over lists of 1 / (k + rank)."""
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [(chunk_map[cid], scores[cid]) for cid in ordered]


class Reranker:
    """
    Scores (query, passage) pairs on a 0-1 relevance scale.
    Uses a cross-encoder when sentence-transformers is installed; otherwise a
    heuristic blend of vector similarity and query-term coverage.
    """

    def __init__(self):
        self._cross_encoder = None
        self._load_failed = False
        self._lock = threading.Lock()

    @property
    def uses_cross_encoder(self) -> bool:
        return settings.ENABLE_RERANKER and CROSS_ENCODER_AVAILABLE and not self._load_failed

    def _get_cross_encoder(self):
        if not self.uses_cross_encoder:
            return None
        with self._lock:
            if self._cross_encoder is None and not self._load_failed:
                try:
                    logger.info(f"Loading cross-encoder {settings.RERANKER_MODEL}...")
                    self._cross_encoder = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
                except Exception as e:
                    logger.warning(f"Could not load cross-encoder ({e}); using heuristic reranking.")
                    self._load_failed = True
        return self._cross_encoder

    def rerank(self, query: str, candidates: List[Dict[str, Any]],
               vector_scores: Optional[Dict[str, float]] = None) -> Tuple[List[Tuple[Dict[str, Any], float]], str]:
        """
        Returns ([(chunk, relevance 0-1)] best-first, scorer name). `candidates` should be in
        fused (RRF) order, which is kept as the tie-breaker for the heuristic scorer.
        """
        if not candidates:
            return [], "none"

        encoder = self._get_cross_encoder()
        if encoder is not None:
            try:
                raw = np.asarray(encoder.predict([(query, c["text"]) for c in candidates],
                                                 show_progress_bar=False), dtype=np.float64).reshape(-1)
                # Depending on the model config, predict() returns raw logits (ms-marco models)
                # or already-sigmoided probabilities; only squash values that are not probabilities.
                probs = raw if raw.min() >= 0.0 and raw.max() <= 1.0 else 1.0 / (1.0 + np.exp(-raw))
                scored = list(zip(candidates, (float(p) for p in probs)))
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored, "cross-encoder"
            except Exception as e:
                logger.warning(f"Cross-encoder reranking failed ({e}); using heuristic reranking.")

        vector_scores = vector_scores or {}
        scored = []
        for rank, chunk in enumerate(candidates):
            coverage = term_coverage(query, chunk["text"] + " " + chunk.get("section", ""))
            cosine = vector_scores.get(chunk["chunk_id"])
            score = coverage if cosine is None else 0.6 * max(cosine, 0.0) + 0.4 * coverage
            # Small rank prior keeps the fused order meaningful for near-ties.
            scored.append((chunk, round(min(score + 0.02 / (rank + 1), 1.0), 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored, "heuristic"


reranker = Reranker()
