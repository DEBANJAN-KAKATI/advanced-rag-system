import re
import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from app.utils.logging import logger

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

class Reranker:
    def __init__(self):
        self._cross_encoder = None

    def _get_cross_encoder(self):
        if self._cross_encoder is None and CROSS_ENCODER_AVAILABLE:
            try:
                logger.info("Loading Cross-Encoder model (ms-marco-MiniLM-L-6-v2)...")
                self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder model: {e}")
        return self._cross_encoder

    def bm25_search(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 15) -> List[Tuple[Dict[str, Any], float]]:
        """Performs BM25 keyword search over provided chunks."""
        if not chunks:
            return []
            
        corpus = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(corpus)
        
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
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combines vector search and BM25 results using Reciprocal Rank Fusion (RRF).
        RRF_score = sum( 1 / (k + rank) )
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
        return [chunk_map[cid] for cid in sorted_cids]

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Applies Cross-Encoder reranking to top candidate chunks.
        Returns sorted list of (chunk, rerank_score).
        """
        if not candidates:
            return []

        encoder = self._get_cross_encoder()
        if encoder:
            try:
                pairs = [(query, c["text"]) for c in candidates]
                scores = encoder.predict(pairs)
                
                # Normalize scores to 0-1 range via sigmoid
                normalized_scores = 1.0 / (1.0 + np.exp(-np.array(scores))) if 'np' in globals() else scores
                
                scored_candidates = list(zip(candidates, [float(s) for s in scores]))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                return scored_candidates[:top_k]
            except Exception as e:
                logger.warning(f"CrossEncoder reranking failed: {e}. Falling back to RRF order.")

        # Fallback scoring: Exact keyword overlap score
        scored = []
        q_words = set(re.findall(r'\w+', query.lower()))
        for c in candidates:
            c_words = set(re.findall(r'\w+', c["text"].lower()))
            overlap = len(q_words.intersection(c_words)) / max(len(q_words), 1)
            scored.append((c, overlap))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

reranker = Reranker()
