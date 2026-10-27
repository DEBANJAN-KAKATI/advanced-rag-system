from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import settings
from app.core.telemetry import stage
from app.retrieval.vectorstore import vector_store
from app.retrieval.query_rewriter import rewrite_query
from app.retrieval.reranker import reranker
from app.utils.logging import logger


@dataclass
class RetrievalConfig:
    """Which retrieval stages run. Defaults reproduce the app's standard pipeline."""
    use_dense: bool = True
    use_bm25: bool = True
    use_query_rewrite: bool = field(default_factory=lambda: settings.ENABLE_QUERY_REWRITE)
    rerank_strategy: str = field(default_factory=lambda: settings.RERANK_STRATEGY)
    candidate_k: int = field(default_factory=lambda: settings.DEFAULT_TOP_K)


def retrieve_relevant_chunks(
    question: str,
    top_k: int = settings.RERANK_TOP_K,
    doc_filter: List[str] = None,
    store=None,
    config: Optional[RetrievalConfig] = None,
) -> Tuple[List[Tuple[Dict[str, Any], float]], List[str]]:
    """
    Executes hybrid RAG retrieval pipeline:
    1. Expands question into query variations
    2. Runs vector similarity search across all query variations
    3. Runs BM25 keyword search across vector store corpus
    4. Merges candidates using Reciprocal Rank Fusion (RRF)
    5. Reranks top candidates with Cross-Encoder to select best top_k chunks
    Returns (scored_chunks, query_expansions)

    `store` and `config` default to the app's global vector store and settings;
    the evaluation harness passes its own to compare strategies.
    """
    store = store or vector_store
    config = config or RetrievalConfig()
    if not (config.use_dense or config.use_bm25):
        raise ValueError("RetrievalConfig must enable dense and/or BM25 search.")

    all_chunks = store.get_all_chunks()
    if not all_chunks:
        logger.warning("No documents in vector store.")
        return [], [question]

    # Filter chunks if doc_filter provided
    if doc_filter:
        all_chunks = [c for c in all_chunks if c.get("doc_id") in doc_filter]

    # 1. Multi-query rewriting
    if config.use_query_rewrite:
        with stage("query_rewrite"):
            expanded_queries = rewrite_query(question)
        logger.info(f"Query expansion produced {len(expanded_queries)} queries: {expanded_queries}")
    else:
        expanded_queries = [question]

    # 2. Vector search across expanded queries
    vector_candidates: List[Tuple[Dict[str, Any], float]] = []
    seen_cids = set()

    if config.use_dense:
        for q in expanded_queries:
            v_results = store.similarity_search(q, top_k=config.candidate_k, doc_filter=doc_filter)
            for chunk, score in v_results:
                cid = chunk["chunk_id"]
                if cid not in seen_cids:
                    vector_candidates.append((chunk, score))
                    seen_cids.add(cid)

    # 3. BM25 keyword search
    bm25_candidates = []
    if config.use_bm25:
        with stage("bm25"):
            bm25_candidates = reranker.bm25_search(question, all_chunks, top_k=config.candidate_k)

    # 4. Reciprocal Rank Fusion
    with stage("fusion"):
        fused = reranker.reciprocal_rank_fusion(vector_candidates, bm25_candidates, with_scores=True)
    fused_candidates = [c for c, _ in fused]

    # 5. Reranking (cross-encoder by default)
    reranked_chunks = reranker.rerank(
        question, fused_candidates, top_k=top_k,
        strategy=config.rerank_strategy, store=store,
        candidate_scores=[s for _, s in fused],
    )

    logger.info(f"Retrieved and reranked {len(reranked_chunks)} chunks for question: '{question}'")
    return reranked_chunks, expanded_queries
