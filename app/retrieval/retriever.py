from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.retrieval.vectorstore import vector_store
from app.retrieval.query_rewriter import rewrite_query
from app.retrieval.reranker import reranker
from app.utils.logging import logger

def retrieve_relevant_chunks(
    question: str,
    top_k: int = settings.RERANK_TOP_K,
    doc_filter: List[str] = None
) -> Tuple[List[Tuple[Dict[str, Any], float]], List[str]]:
    """
    Executes hybrid RAG retrieval pipeline:
    1. Expands question into query variations
    2. Runs vector similarity search across all query variations
    3. Runs BM25 keyword search across vector store corpus
    4. Merges candidates using Reciprocal Rank Fusion (RRF)
    5. Reranks top candidates with Cross-Encoder to select best top_k chunks
    Returns (scored_chunks, query_expansions)
    """
    all_chunks = vector_store.get_all_chunks()
    if not all_chunks:
        logger.warning("No documents in vector store.")
        return [], [question]

    # Filter chunks if doc_filter provided
    if doc_filter:
        all_chunks = [c for c in all_chunks if c.get("doc_id") in doc_filter]

    # 1. Multi-query rewriting
    expanded_queries = rewrite_query(question)
    logger.info(f"Query expansion produced {len(expanded_queries)} queries: {expanded_queries}")

    # 2. Vector search across expanded queries
    vector_candidates: List[Tuple[Dict[str, Any], float]] = []
    seen_cids = set()
    
    for q in expanded_queries:
        v_results = vector_store.similarity_search(q, top_k=settings.DEFAULT_TOP_K, doc_filter=doc_filter)
        for chunk, score in v_results:
            cid = chunk["chunk_id"]
            if cid not in seen_cids:
                vector_candidates.append((chunk, score))
                seen_cids.add(cid)

    # 3. BM25 keyword search
    bm25_candidates = reranker.bm25_search(question, all_chunks, top_k=settings.DEFAULT_TOP_K)

    # 4. Reciprocal Rank Fusion
    fused_candidates = reranker.reciprocal_rank_fusion(vector_candidates, bm25_candidates)

    # 5. Cross-Encoder Reranking
    reranked_chunks = reranker.rerank(question, fused_candidates, top_k=top_k)

    logger.info(f"Retrieved and reranked {len(reranked_chunks)} chunks for question: '{question}'")
    return reranked_chunks, expanded_queries
