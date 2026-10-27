import numpy as np
from typing import List
from app.core.config import settings
from app.core.pricing import estimate_tokens
from app.core.telemetry import record_embedding
from app.utils.logging import logger

# Try Google GenAI SDKs first
try:
    from google import genai
    GENAI_NEW_SDK = True
except ImportError:
    GENAI_NEW_SDK = False

# Local sentence-transformers fallback
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDER_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDER_AVAILABLE = False

class Embedder:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.EMBEDDING_MODEL
        self._local_model = None
        self.client = None
        
        if settings.has_llm_key:
            if GENAI_NEW_SDK:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                    logger.info("Initialized Google GenAI Embedder client.")
                except Exception as e:
                    logger.warning(f"Could not init GenAI Client: {e}")
                    self.client = None

    def _get_local_model(self):
        if self._local_model is None and LOCAL_EMBEDDER_AVAILABLE:
            logger.info("Loading local sentence-transformers model (all-MiniLM-L6-v2)...")
            self._local_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._local_model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings into numpy vector array."""
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        # 1. Try Gemini API via new SDK
        if GENAI_NEW_SDK and hasattr(self, 'client') and self.client:
            try:
                embeddings = []
                # Batch embed
                for text in texts:
                    res = self.client.models.embed_content(
                        model=self.model_name,
                        contents=text,
                    )
                    vals = res.embeddings[0].values if hasattr(res, 'embeddings') and res.embeddings else res.embedding.values
                    embeddings.append(vals)
                record_embedding("gemini", self.model_name, len(texts), sum(estimate_tokens(t) for t in texts))
                arr = np.array(embeddings, dtype=np.float32)
                # Normalize L2
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return arr / norms
            except Exception as e:
                logger.warning(f"Gemini embedding via GenAI SDK failed: {type(e).__name__} - {e}. Trying legacy...")

        # Fallback to local sentence transformer
        local_model = self._get_local_model()
        if local_model:
            logger.info("Generating embeddings using local sentence-transformers fallback.")
            embeddings = local_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            record_embedding("sentence-transformers", "all-MiniLM-L6-v2", len(texts), sum(estimate_tokens(t) for t in texts))
            return embeddings.astype(np.float32)

        raise RuntimeError("No embedding provider available! Please verify GEMINI_API_KEY or sentence-transformers installation.")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query string."""

        arr = self.embed_texts([query])
        return arr

embedder = Embedder()
