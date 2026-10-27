import threading
from typing import List, Optional
import numpy as np
from app.core.config import settings
from app.utils.logging import logger
from app.utils.retry import call_with_retry, describe_error

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDER_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDER_AVAILABLE = False


class EmbeddingError(RuntimeError):
    """Raised when texts cannot be embedded. Never silently falls back to another model."""


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class Embedder:
    """
    Base embedder. Every vector in an index must come from the same model, so each
    embedder exposes a `signature`; the vector store re-embeds when it changes
    instead of mixing vector spaces.
    """
    signature = "none"
    description = "none"

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class GeminiEmbedder(Embedder):
    BATCH_SIZE = 100  # Gemini batch embedding limit per request

    def __init__(self, api_key: str, model: str):
        if not GENAI_AVAILABLE:
            raise EmbeddingError("google-genai is not installed (pip install google-genai).")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.signature = f"gemini:{model}"
        self.description = f"Gemini {model}"

    def _embed(self, texts: List[str], task_type: str) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start:start + self.BATCH_SIZE]
            try:
                res = call_with_retry(
                    lambda: self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=genai_types.EmbedContentConfig(task_type=task_type),
                    ),
                    what="Gemini embedding",
                    max_total_wait=max(settings.LLM_MAX_RETRY_WAIT_SECONDS, 30),
                )
            except Exception as e:
                raise EmbeddingError(f"Gemini embedding failed: {describe_error(e)}") from e
            if not res.embeddings or len(res.embeddings) != len(batch):
                raise EmbeddingError("Gemini embedding returned an unexpected number of vectors.")
            vectors.extend(e.values for e in res.embeddings)
        return _normalize(np.array(vectors, dtype=np.float32))

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_QUERY")


class LocalEmbedder(Embedder):
    """sentence-transformers model running on this machine (no API quota, works offline)."""

    def __init__(self, model: str):
        if not LOCAL_EMBEDDER_AVAILABLE:
            raise EmbeddingError(
                "Local embeddings need sentence-transformers (pip install sentence-transformers), "
                "or set GEMINI_API_KEY to use Gemini embeddings."
            )
        self.model_name = model
        self.signature = f"local:{model}"
        self.description = f"Local {model}"
        self._model = None
        self._lock = threading.Lock()
        # BGE models retrieve better when short queries carry this instruction prefix.
        self.query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model.lower() else ""

    def _get_model(self):
        with self._lock:
            if self._model is None:
                logger.info(f"Loading local embedding model {self.model_name} (first use may download it)...")
                try:
                    self._model = SentenceTransformer(self.model_name)
                except Exception as e:
                    raise EmbeddingError(f"Could not load local embedding model '{self.model_name}': {e}") from e
            return self._model

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._get_model().encode(texts, batch_size=32, convert_to_numpy=True,
                                           normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])


def create_embedder() -> Optional[Embedder]:
    """
    Builds the configured embedder. Returns None when no embedding backend is usable,
    in which case the system runs keyword-only (BM25) retrieval.
    """
    provider = settings.EMBEDDING_PROVIDER
    try:
        if provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise EmbeddingError("EMBEDDING_PROVIDER=gemini but no GEMINI_API_KEY is set.")
            return GeminiEmbedder(settings.GEMINI_API_KEY, settings.EMBEDDING_MODEL)
        if provider == "local":
            return LocalEmbedder(settings.LOCAL_EMBEDDING_MODEL)
    except EmbeddingError as e:
        logger.warning(f"{e} Falling back to keyword-only retrieval.")
    return None
