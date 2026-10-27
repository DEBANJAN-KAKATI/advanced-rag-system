import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from app.core.config import settings
from app.core.schemas import AnswerStyle
from app.core.telemetry import stage
from app.retrieval.embedder import EmbeddingError
from app.retrieval.keyword import tokenize
from app.retrieval.query_rewriter import rewrite_query
from app.retrieval.reranker import reciprocal_rank_fusion, reranker
from app.retrieval.vectorstore import vector_store
from app.utils.logging import logger

_SUMMARY_RE = re.compile(
    r"\b(summari[sz]e|summary|overview|tl;?dr|gist|main (?:points?|ideas?|takeaways?)|key (?:points?|takeaways?)|"
    r"what(?:'s| is) (?:this|the|it) (?:\w+ )?about)\b",
    re.IGNORECASE,
)
_GENERIC_WORDS = {"report", "paper", "pdf", "book", "text", "article", "entire", "whole", "all", "main", "key",
                  "point", "uploaded", "content", "contents", "brief", "short", "quick", "about", "doc"}


@dataclass
class RetrievalConfig:
    """Which retrieval stages run. Defaults reproduce the app's configured pipeline."""
    use_dense: bool = True
    use_bm25: bool = True
    use_query_rewrite: bool = field(default_factory=lambda: settings.QUERY_REWRITE)
    rerank_strategy: str = field(default_factory=lambda: settings.RERANK_STRATEGY)
    candidate_k: int = field(default_factory=lambda: settings.CANDIDATE_TOP_K)


@dataclass
class RetrievalResult:
    chunks: List[Tuple[Dict[str, Any], float]]
    standalone_question: str
    queries: List[str]
    mode: str = "hybrid"          # hybrid | dense | keyword | summary
    scorer: str = "keyword"       # the rerank strategy that actually ran
    low_relevance: bool = False
    notices: List[str] = field(default_factory=list)


def top_k_for_style(style: AnswerStyle) -> int:
    return {
        AnswerStyle.CONCISE: settings.TOP_K_CONCISE,
        AnswerStyle.MEDIUM: settings.TOP_K_MEDIUM,
        AnswerStyle.LONG: settings.TOP_K_LONG,
    }.get(style, settings.TOP_K_MEDIUM)


def is_summary_request(question: str) -> bool:
    """True for whole-document requests ("summarize this", "what is this report about?")."""
    if not _SUMMARY_RE.search(question):
        return False
    remaining = [t for t in tokenize(_SUMMARY_RE.sub(" ", question)) if t not in _GENERIC_WORDS]
    return len(remaining) == 0


def _summary_chunks(store, doc_filter: Optional[List[str]]) -> List[Tuple[Dict[str, Any], float]]:
    """
    Similarity search cannot answer "summarize the document", so pick passages spread
    evenly through each document in scope, within the summary character budget.
    """
    chunks = store.get_all_chunks()
    doc_ids = list(dict.fromkeys(c["doc_id"] for c in chunks if not doc_filter or c["doc_id"] in doc_filter))
    if not doc_ids:
        return []
    per_doc_budget = settings.SUMMARY_CONTEXT_CHARS // len(doc_ids)
    selected: List[Tuple[Dict[str, Any], float]] = []
    for doc_id in doc_ids:
        doc_chunks = [c for c in chunks if c["doc_id"] == doc_id]
        avg = max(sum(c["char_count"] for c in doc_chunks) / len(doc_chunks), 1)
        n = max(1, min(len(doc_chunks), int(per_doc_budget // avg)))
        step = len(doc_chunks) / n
        picks = sorted({int(i * step) for i in range(n)})
        selected.extend((doc_chunks[i], 1.0) for i in picks)
    return selected


def retrieve_relevant_chunks(
    question: str,
    style: AnswerStyle = AnswerStyle.MEDIUM,
    doc_filter: Optional[List[str]] = None,
    chat_history: str = "",
    top_k: Optional[int] = None,
    store=None,
    config: Optional[RetrievalConfig] = None,
) -> RetrievalResult:
    """
    Hybrid retrieval pipeline:
    1. Rewrite the question into a standalone query (+ alternatives) using the conversation
    2. Vector search (all queries embedded in one call) and BM25 keyword search per query
    3. Reciprocal Rank Fusion of all ranked lists
    4. Rerank fused candidates and, with the cross-encoder, drop passages below the
       relevance threshold

    `store` and `config` default to the app's global index and settings; the evaluation
    harness passes its own to compare strategies.
    """
    store = store or vector_store
    config = config or RetrievalConfig()
    if not (config.use_dense or config.use_bm25):
        raise ValueError("RetrievalConfig must enable dense and/or BM25 search.")
    top_k = top_k or top_k_for_style(style)
    doc_filter = doc_filter or None

    if config.use_query_rewrite or chat_history:
        with stage("query_rewrite"):
            standalone, queries = rewrite_query(question, chat_history, expand=config.use_query_rewrite)
    else:
        standalone, queries = question.strip(), [question.strip()]

    if is_summary_request(standalone):
        chunks = _summary_chunks(store, doc_filter)
        logger.info(f"Summary request: using {len(chunks)} passages spread across the document(s).")
        return RetrievalResult(chunks, standalone, queries, mode="summary", scorer="none")

    notices: List[str] = []
    vector_lists: List[List[Tuple[Dict[str, Any], float]]] = []
    if config.use_dense and store.embedder is not None:
        if not store.vectors_ready:
            store.ensure_index()
        if store.vectors_ready:
            try:
                with stage("embed_query"):
                    query_vectors = store.embedder.embed_queries(queries)
                vector_lists = store.vector_search(query_vectors, config.candidate_k, doc_filter)
            except EmbeddingError as e:
                logger.warning(f"Query embedding failed, using keyword search only: {e}")
                notices.append("Semantic search was unavailable for this question, so keyword search was used.")

    keyword_lists = []
    if config.use_bm25:
        with stage("bm25"):
            keyword_lists = [store.keyword_search(q, config.candidate_k, doc_filter) for q in queries]
    mode = "hybrid" if vector_lists and keyword_lists else ("dense" if vector_lists else "keyword")

    with stage("fusion"):
        fused = reciprocal_rank_fusion(vector_lists + keyword_lists)[:config.candidate_k]
    if not fused:
        return RetrievalResult([], standalone, queries, mode=mode, notices=notices)

    reranked, scorer = reranker.rerank(standalone, [c for c, _ in fused],
                                       strategy=config.rerank_strategy, store=store)

    low_relevance = False
    if scorer == "cross-encoder":
        relevant = [(c, s) for c, s in reranked if s >= settings.MIN_RELEVANCE_SCORE]
        if not relevant:
            # Nothing clears the bar: pass a few best passages and let the model decide.
            low_relevance = True
            relevant = reranked[:min(3, top_k)]
    else:
        relevant = reranked

    selected = relevant[:top_k]
    logger.info(f"Retrieved {len(selected)} passages ({mode}, {scorer}) for: {standalone!r}")
    return RetrievalResult(selected, standalone, queries, mode=mode, scorer=scorer,
                           low_relevance=low_relevance, notices=notices)
