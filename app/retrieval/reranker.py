import threading
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from app.core.config import settings
from app.core.telemetry import annotate, stage
from app.retrieval.keyword import term_coverage
from app.utils.logging import logger

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

# Selectable via settings.RERANK_STRATEGY; compared in docs/EVALUATION_REPORT.md.
RERANK_STRATEGIES = ("cross-encoder", "keyword", "none", "mmr", "sentence-maxsim")

Scored = List[Tuple[Dict[str, Any], float]]


def reciprocal_rank_fusion(ranked_lists: List[Scored], k: int = 60) -> Scored:
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
    Reorders fused candidates and scores them on a 0-1 relevance scale.

    Strategies:
      - cross-encoder:   ms-marco MiniLM cross-encoder (default); falls back to keyword
                         when sentence-transformers or the model is unavailable
      - keyword:         share of query terms present in the passage
      - none:            keep the fused (RRF) order
      - mmr:             maximal marginal relevance over stored vectors
      - sentence-maxsim: best cosine between the query and any single sentence of the
                         passage; embeds candidate sentences per query, so slow on CPU
    The strategy that actually ran is returned and recorded as the
    `rerank_strategy_used` trace annotation, so fallbacks are never silent.
    """

    def __init__(self):
        self._cross_encoder = None
        self._load_failed = False
        self._lock = threading.Lock()

    @property
    def uses_cross_encoder(self) -> bool:
        return settings.RERANK_STRATEGY == "cross-encoder" and CROSS_ENCODER_AVAILABLE and not self._load_failed

    def _get_cross_encoder(self):
        if not CROSS_ENCODER_AVAILABLE or self._load_failed:
            return None
        with self._lock:
            if self._cross_encoder is None and not self._load_failed:
                try:
                    logger.info(f"Loading cross-encoder {settings.RERANKER_MODEL}...")
                    self._cross_encoder = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
                except Exception as e:
                    # Remember the failure so every query doesn't retry the download.
                    logger.warning(f"Could not load cross-encoder ({e}); using keyword reranking.")
                    self._load_failed = True
        return self._cross_encoder

    def rerank(self, query: str, candidates: List[Dict[str, Any]],
               strategy: Optional[str] = None, store=None) -> Tuple[Scored, str]:
        """
        Returns ([(chunk, relevance 0-1)] best-first, strategy actually used).
        `candidates` must be in fused (RRF) order, which breaks ties.
        Strategies that need vectors (mmr, sentence-maxsim) fall back to keyword
        when `store` has no usable embedding index.
        """
        if not candidates:
            return [], "none"
        strategy = strategy or settings.RERANK_STRATEGY
        if strategy not in RERANK_STRATEGIES:
            raise ValueError(f"Unknown rerank strategy {strategy!r}; expected one of {RERANK_STRATEGIES}")

        with stage("rerank"):
            scored = None
            if strategy == "cross-encoder":
                scored = self._cross_encoder_scores(query, candidates)
            elif strategy == "mmr":
                scored = self._mmr(query, candidates, store)
            elif strategy == "sentence-maxsim":
                scored = self._sentence_maxsim(query, candidates, store)
            elif strategy == "none":
                scored = [(c, round(term_coverage(query, c["text"]), 4)) for c in candidates]
            elif strategy == "keyword":
                scored = self._keyword(query, candidates)

            used = strategy
            if scored is None:
                scored, used = self._keyword(query, candidates), "keyword"
        annotate("rerank_strategy_used", used)
        return scored, used

    def _cross_encoder_scores(self, query: str, candidates: List[Dict[str, Any]]) -> Optional[Scored]:
        encoder = self._get_cross_encoder()
        if encoder is None:
            return None
        try:
            raw = np.asarray(encoder.predict([(query, c["text"]) for c in candidates],
                                             show_progress_bar=False), dtype=np.float64).reshape(-1)
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed ({e}); using keyword reranking.")
            return None
        # Depending on the model config, predict() returns raw logits (ms-marco models)
        # or already-sigmoided probabilities; only squash values that are not probabilities.
        probs = raw if raw.min() >= 0.0 and raw.max() <= 1.0 else 1.0 / (1.0 + np.exp(-raw))
        scored = list(zip(candidates, (float(p) for p in probs)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _keyword(query: str, candidates: List[Dict[str, Any]]) -> Scored:
        scored = []
        for rank, chunk in enumerate(candidates):
            coverage = term_coverage(query, chunk["text"] + " " + chunk.get("section", ""))
            # Small rank prior keeps the fused order meaningful for near-ties.
            scored.append((chunk, round(min(coverage + 0.02 / (rank + 1), 1.0), 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _vector_store(store):
        if store is None:
            from app.retrieval.vectorstore import vector_store as store
        return store if store.embedder is not None and store.vectors_ready else None

    def _mmr(self, query: str, candidates: List[Dict[str, Any]], store, lambda_mult: float = 0.7) -> Optional[Scored]:
        store = self._vector_store(store)
        if store is None:
            return None
        q = store.embedder.embed_queries([query])[0]
        vecs = store.get_vectors([c["chunk_id"] for c in candidates])
        relevance = vecs @ q
        selected: List[int] = []
        remaining = list(range(len(candidates)))
        while remaining:
            redundancy = np.max(vecs[remaining] @ vecs[selected].T, axis=1) if selected else np.zeros(len(remaining))
            mmr = lambda_mult * relevance[remaining] - (1 - lambda_mult) * redundancy
            best = remaining[int(np.argmax(mmr))]
            selected.append(best)
            remaining.remove(best)
        return [(candidates[i], float(max(relevance[i], 0.0))) for i in selected]

    def _sentence_maxsim(self, query: str, candidates: List[Dict[str, Any]], store) -> Optional[Scored]:
        from app.retrieval.chunker import sentence_spans
        store = self._vector_store(store)
        if store is None:
            return None
        sentences, owners = [], []
        for i, c in enumerate(candidates):
            for _, sentence in sentence_spans(c["text"]):
                if len(sentence) >= 20:
                    sentences.append(sentence)
                    owners.append(i)
        if not sentences:
            return None
        q = store.embedder.embed_queries([query])[0]
        sims = store.embedder.embed_documents(sentences) @ q
        best = np.zeros(len(candidates))
        for owner, sim in zip(owners, sims):
            best[owner] = max(best[owner], float(sim))
        order = np.argsort(-best, kind="stable")
        return [(candidates[i], float(best[i])) for i in order]


reranker = Reranker()
