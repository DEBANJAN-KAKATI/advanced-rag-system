import os
from pathlib import Path
import sys
from dotenv import load_dotenv

class Settings:
    def __init__(self):
        load_dotenv()

        if getattr(sys, 'frozen', False):
            self.BASE_DIR = Path(sys.executable).parent
        else:
            self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.VECTOR_STORE_DIR = self.DATA_DIR / "vectorstore"
        self.UPLOAD_DIR = self.DATA_DIR / "uploads"
        
        # Ensure directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        
        # Load API key from api_key_here file if it exists
        api_key_file = self.BASE_DIR / "api_key_here"
        file_api_key = None
        if api_key_file.exists():
            try:
                # Read lines, strip whitespace, ignore comments
                for line in api_key_file.read_text().splitlines():
                    clean_line = line.strip()
                    if clean_line and not clean_line.startswith("#"):
                        file_api_key = clean_line
                        break
            except Exception:
                pass
                
        self.GEMINI_API_KEY = file_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "your_api_key"
        os.environ["GEMINI_API_KEY"] = self.GEMINI_API_KEY
        os.environ["GOOGLE_API_KEY"] = self.GEMINI_API_KEY
        self.LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").replace("models/", "")
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
        self.DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "15"))
        self.RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

        # Retrieval strategy knobs (see evaluation/ and docs/EVALUATION_REPORT.md)
        # CHUNK_STRATEGY: structure | fixed | sentence | semantic
        self.CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "structure")
        # RERANK_STRATEGY: cross-encoder | keyword | none | mmr | sentence-maxsim
        self.RERANK_STRATEGY = os.getenv("RERANK_STRATEGY", "cross-encoder")
        self.ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() in ("1", "true", "yes")

    @property
    def has_llm_key(self) -> bool:
        """True only for a real key, not the "your_api_key" placeholder."""
        return bool(self.GEMINI_API_KEY) and self.GEMINI_API_KEY != "your_api_key"

settings = Settings()
