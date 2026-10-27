import os
import sys
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    def load_dotenv(*_args, **_kwargs):
        return False


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _read_key_file(path: Path) -> Optional[str]:
    """Reads the first non-comment, non-blank line of the legacy `api_key_here` file."""
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            clean_line = line.strip().strip('"').strip("'")
            if clean_line and not clean_line.startswith("#"):
                return clean_line
    except OSError:
        pass
    return None


class Settings:
    def __init__(self):
        if getattr(sys, "frozen", False):
            self.BASE_DIR = Path(sys.executable).parent
        else:
            self.BASE_DIR = Path(__file__).resolve().parent.parent.parent

        load_dotenv(self.BASE_DIR / ".env")

        # --- Storage ---
        self.DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(self.BASE_DIR / "data"))).resolve()
        self.VECTOR_STORE_DIR = self.DATA_DIR / "vectorstore"
        self.UPLOAD_DIR = self.DATA_DIR / "uploads"
        for d in (self.DATA_DIR, self.VECTOR_STORE_DIR, self.UPLOAD_DIR):
            d.mkdir(parents=True, exist_ok=True)

        # --- Server ---
        self.HOST = os.getenv("HOST", "127.0.0.1")
        self.PORT = _env_int("PORT", 8000)
        self.OPEN_BROWSER = _env_bool("OPEN_BROWSER", True)
        self.CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
        self.MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 50)

        # --- Credentials ---
        # Priority: environment / .env, then the legacy `api_key_here` file.
        self.GEMINI_API_KEY: Optional[str] = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or _read_key_file(self.BASE_DIR / "api_key_here")
            or None
        )
        self.OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY") or None
        # Any OpenAI-compatible server works: OpenAI, Ollama (http://localhost:11434/v1), LM Studio, vLLM...
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

        # --- LLM ---
        # "gemini" | "openai" | "none" | "auto" (gemini if a Gemini key exists, else openai if configured)
        self.LLM_PROVIDER = self._resolve_llm_provider(os.getenv("LLM_PROVIDER", "auto").strip().lower())
        default_model = "gemini-flash-latest" if self.LLM_PROVIDER == "gemini" else "gpt-4o-mini"
        self.LLM_MODEL = os.getenv("LLM_MODEL", default_model).replace("models/", "")
        self.LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.2)
        self.LLM_TIMEOUT_SECONDS = _env_int("LLM_TIMEOUT_SECONDS", 120)
        # Upper bound on total time spent waiting between retries on rate-limit errors.
        self.LLM_MAX_RETRY_WAIT_SECONDS = _env_int("LLM_MAX_RETRY_WAIT_SECONDS", 20)
        self.QUERY_REWRITE = _env_bool("QUERY_REWRITE", True)
        self.SUGGEST_FOLLOWUPS = _env_bool("SUGGEST_FOLLOWUPS", True)

        # --- Embeddings ---
        # "gemini" | "local" | "none" | "auto" (gemini if a key exists, otherwise local sentence-transformers)
        self.EMBEDDING_PROVIDER = self._resolve_embedding_provider(
            os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()
        )
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").replace("models/", "")
        self.LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

        # --- Reranking ---
        self.ENABLE_RERANKER = _env_bool("ENABLE_RERANKER", True)
        self.RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        # Cross-encoder relevance (0-1) below which a passage is treated as unrelated.
        self.MIN_RELEVANCE_SCORE = _env_float("MIN_RELEVANCE_SCORE", 0.02)

        # --- Chunking & retrieval ---
        # Sizes are in characters (~4 characters per token for English text).
        self.CHUNK_SIZE = _env_int("CHUNK_SIZE", 1200)
        self.CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 200)
        self.CANDIDATE_TOP_K = _env_int("CANDIDATE_TOP_K", _env_int("DEFAULT_TOP_K", 25))
        # Context size per answer style; overrides with RERANK_TOP_K apply to "medium".
        self.TOP_K_CONCISE = _env_int("TOP_K_CONCISE", 4)
        self.TOP_K_MEDIUM = _env_int("TOP_K_MEDIUM", _env_int("RERANK_TOP_K", 6))
        self.TOP_K_LONG = _env_int("TOP_K_LONG", 10)
        # Character budget for whole-document summary questions.
        self.SUMMARY_CONTEXT_CHARS = _env_int("SUMMARY_CONTEXT_CHARS", 24000)

        # --- Conversation memory ---
        self.MEMORY_TURNS = _env_int("MEMORY_TURNS", 6)
        self.MAX_CONVERSATIONS = _env_int("MAX_CONVERSATIONS", 200)

    def _resolve_llm_provider(self, requested: str) -> str:
        if requested in ("gemini", "openai", "none"):
            return requested
        if self.GEMINI_API_KEY:
            return "gemini"
        if self.OPENAI_API_KEY or os.getenv("OPENAI_BASE_URL"):
            return "openai"
        return "none"

    def _resolve_embedding_provider(self, requested: str) -> str:
        if requested in ("gemini", "local", "none"):
            return requested
        return "gemini" if self.GEMINI_API_KEY else "local"


settings = Settings()
