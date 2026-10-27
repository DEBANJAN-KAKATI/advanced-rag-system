from fastapi import APIRouter
from app.core.config import settings
from app.core.schemas import StatusResponse
from app.generation import llm as llm_module
from app.retrieval.reranker import reranker, CROSS_ENCODER_AVAILABLE
from app.retrieval.vectorstore import vector_store
from app.utils.filetypes import supported_extensions_label

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/status", response_model=StatusResponse)
def status():
    """Reports which components are active, so the UI can explain degraded modes."""
    client = llm_module.llm
    warnings = []
    if client is None:
        warnings.append("No LLM configured: answers are extracted sentences, not AI-written. "
                        "Set GEMINI_API_KEY (or an OpenAI-compatible server) in .env.")
    if vector_store.embedder is None:
        warnings.append("No embedding model available: using keyword search only. "
                        "Set GEMINI_API_KEY or install sentence-transformers.")
    elif not vector_store.vectors_ready and vector_store.chunks:
        warnings.append(f"Semantic index not ready: {vector_store.last_index_error or 'building...'}")

    if not settings.ENABLE_RERANKER:
        reranker_name = "disabled"
    elif CROSS_ENCODER_AVAILABLE and reranker.uses_cross_encoder:
        reranker_name = settings.RERANKER_MODEL
    else:
        reranker_name = "heuristic"

    return StatusResponse(
        llm_provider=client.provider if client else "none",
        llm_model=client.model if client else None,
        llm_available=client is not None,
        embedding_provider=vector_store.embedder.description if vector_store.embedder else "none (keyword only)",
        vectors_ready=vector_store.vectors_ready,
        reranker=reranker_name,
        documents=len(vector_store.documents),
        chunks=len(vector_store.chunks),
        max_upload_mb=settings.MAX_UPLOAD_MB,
        supported_formats=supported_extensions_label(),
        warnings=warnings,
    )
