import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from app.core.config import settings
from app.core.telemetry import annotate, stage
from app.utils.logging import logger

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

# Strategies selectable via settings.RERANK_STRATEGY or the `strategy` argument.
RERANK_STRATEGIES = ("cross-encoder", "keyword", "none", "mmr", "sentence-maxsim")


class Reranker:
    def __init__(self):
        self._cross_encoder = None
        self._cross_encoder_failed = False
        self._bm25_cache_key = None
        self._bm25_cache = None
        self.cache_bm25 = True  # the evaluation harness turns this off to measure the uncached cost

    def _get_cross_encoder(self):
        if self._cross_encoder is None and CROSS_ENCODER_AVAILABLE and not self._cross_encoder_failed:
            try:
                logger.info("Loading Cross-Encoder model (ms-marco-MiniLM-L-6-v2)...")
                self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                # Remember the failure so every query doesn't retry the download.
                self._cross_encoder_failed = True
                logger.warning(f"Could not load CrossEncoder model: {e}")
        return self._cross_encoder

    def _get_bm25(self, chunks: List[Dict[str, Any]]) -> BM25Okapi:
        """Builds the BM25 index once per distinct chunk set instead of once per query."""
        key = (len(chunks), tuple(c["chunk_id"] for c in chunks))
        if key != self._bm25_cache_key or not self.cache_bm25:
            corpus = [c["text"].lower().split() for c in chunks]
            self._bm25_cache = BM25Okapi(corpus)
            self._bm25_cache_key = key
        return self._bm25_cache

    def bm25_search(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 15) -> List[Tuple[Dict[str, Any], float]]:
        """Performs BM25 keyword search over provided chunks."""
        if not chunks:
            return []

        bm25 = self._get_bm25(chunks)

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score > 0:
                results.append((chunks[idx], float(score)))

        return results

    def reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[Dict[str, Any], float]],
        bm25_results: List[Tuple[Dict[str, Any], float]],
        k: int = 60,
        with_scores: bool = False,
    ) -> List[Any]:
        """
        Combines vector search and BM25 results using Reciprocal Rank Fusion (RRF).
        RRF_score = sum( 1 / (k + rank) )
        Returns chunks, or (chunk, rrf_score) pairs when with_scores=True.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Process vector ranks
        for rank, (chunk, score) in enumerate(vector_results):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))

        # Process BM25 ranks
        for rank, (chunk, score) in enumerate(bm25_results):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))

        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        if with_scores:
            return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_cids]
        return [chunk_map[cid] for cid in sorted_cids]

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        strategy: Optional[str] = None,
        store=None,
        embedder=None,
        candidate_scores: Optional[List[float]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Reorders fused candidates and returns the best top_k as (chunk, score).

        Strategies:
          - cross-encoder:   ms-marco MiniLM cross-encoder; falls back to keyword
                             overlap if the model cannot be loaded (default)
          - keyword:         fraction of query words present in the chunk
          - none:            keep the fused (RRF) order
          - mmr:             maximal marginal relevance over stored chunk vectors,
                             trading relevance against redundancy (needs `store`)
          - sentence-maxsim: best cosine between the query and any single sentence
                             of the chunk. Embeds candidate sentences per query, so
                             only use it with a local embedder.
        The strategy that actually ran is recorded as the `rerank_strategy_used`
        trace annotation, so silent fallbacks show up in evaluation results.
        """
        if not candidates:
            return []
        strategy = strategy or settings.RERANK_STRATEGY

        with stage("rerank"):
            if strategy == "cross-encoder":
                encoder = self._get_cross_encoder()
                if encoder:
                    try:
                        pairs = [(query, c["text"]) for c in candidates]
                        scores = encoder.predict(pairs)
                        scored_candidates = list(zip(candidates, [float(s) for s in scores]))
                        scored_candidates.sort(key=lambda x: x[1], reverse=True)
                        annotate("rerank_strategy_used", "cross-encoder")
                        return scored_candidates[:top_k]
                    except Exception as e:
                        logger.warning(f"CrossEncoder reranking failed: {e}. Falling back to keyword overlap.")
                strategy = "keyword"

            if strategy == "none":
                annotate("rerank_strategy_used", "none")
                scores = candidate_scores or [1.0 / (rank + 1) for rank in range(len(candidates))]
                return list(zip(candidates, scores))[:top_k]

            if strategy == "mmr":
                annotate("rerank_strategy_used", "mmr")
                return self._mmr(query, candidates, top_k, store, embedder)

            if strategy == "sentence-maxsim":
                annotate("rerank_strategy_used", "sentence-maxsim")
                return self._sentence_maxsim(query, candidates, top_k, embedder or getattr(store, "embedder", None))

            if strategy != "keyword":
                raise ValueError(f"Unknown rerank strategy {strategy!r}; expected one of {RERANK_STRATEGIES}")

            # Keyword overlap score
            annotate("rerank_strategy_used", "keyword")
            scored = []
            q_words = set(re.findall(r'\w+', query.lower()))
            for c in candidates:
                c_words = set(re.findall(r'\w+', c["text"].lower()))
                overlap = len(q_words.intersection(c_words)) / max(len(q_words), 1)
                scored.append((c, overlap))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    def _mmr(self, query, candidates, top_k, store, embedder, lambda_mult: float = 0.7):
        if store is None:
            from app.retrieval.vectorstore import vector_store as store
        emb = embedder or store.embedder
        q = emb.embed_query(query)[0]
        vecs = store.get_vectors([c["chunk_id"] for c in candidates])
        relevance = vecs @ q
        selected: List[int] = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected) < top_k:
            if selected:
                redundancy = np.max(vecs[remaining] @ vecs[selected].T, axis=1)
            else:
                redundancy = np.zeros(len(remaining))
            mmr = lambda_mult * relevance[remaining] - (1 - lambda_mult) * redundancy
            best = remaining[int(np.argmax(mmr))]
            selected.append(best)
            remaining.remove(best)
        return [(candidates[i], float(relevance[i])) for i in selected]

    def _sentence_maxsim(self, query, candidates, top_k, embedder):
        if embedder is None:
            from app.retrieval.embedder import embedder
        from app.retrieval.chunker import split_sentences
        sentences, owners = [], []
        for i, c in enumerate(candidates):
            for _, s in split_sentences(c["text"]):
                if len(s) >= 20:
                    sentences.append(s)
                    owners.append(i)
        if not sentences:
            return [(c, 0.0) for c in candidates[:top_k]]
        q = embedder.embed_query(query)[0]
        sims = embedder.embed_texts(sentences) @ q
        best = np.full(len(candidates), -1.0)
        for owner, sim in zip(owners, sims):
            best[owner] = max(best[owner], float(sim))
        order = np.argsort(-best, kind="stable")[:top_k]
        return [(candidates[i], float(best[i])) for i in order]


reranker = Reranker()
