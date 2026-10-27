"""
Test configuration: every test runs offline against a temporary data directory,
with no LLM, no real embedding model and no cross-encoder.
"""
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

# Must happen before any `app.*` import: settings are read at import time.
os.environ["RAG_DATA_DIR"] = tempfile.mkdtemp(prefix="rag-tests-")
os.environ["LLM_PROVIDER"] = "none"
os.environ["EMBEDDING_PROVIDER"] = "none"
os.environ["ENABLE_RERANKER"] = "false"
for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
    os.environ.pop(key, None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.telemetry import record_llm_call  # noqa: E402
from app.retrieval.embedder import Embedder  # noqa: E402
from app.retrieval.vectorstore import VectorStore  # noqa: E402


class FakeEmbedder(Embedder):
    """Deterministic hashed bag-of-words embeddings; counts how many texts it embedded."""

    def __init__(self, dim: int = 64, name: str = "fake"):
        self.dim = dim
        self.signature = f"test:{name}:{dim}"
        self.description = f"Fake {name}"
        self.embedded_texts = 0
        self.fail = False

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        from app.retrieval.embedder import EmbeddingError
        if self.fail:
            raise EmbeddingError("simulated embedding outage")
        self.embedded_texts += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_queries(self, texts):
        return self.embed_documents(texts)


class FakeLLM:
    """Records prompts (and telemetry, like the real clients) and streams a canned answer."""
    provider = "fake"
    model = "fake-model"

    def __init__(self, answer: str = "", fail: Exception = None, rewrite: str = None):
        self.answer = answer
        self.fail = fail
        self.rewrite = rewrite
        self.prompts = []

    def generate(self, system, prompt, json_mode=False, fast=False, purpose="llm"):
        self.prompts.append(prompt)
        if self.fail:
            raise self.fail
        record_llm_call(purpose, self.model, len(prompt) // 4, 10, estimated=True, provider=self.provider)
        return self.rewrite or '{"standalone": "", "queries": []}'

    def stream(self, system, prompt, purpose="answer"):
        self.prompts.append(prompt)
        if self.fail:
            raise self.fail
        for i in range(0, len(self.answer), 5):
            yield self.answer[i:i + 5]
        record_llm_call(purpose, self.model, len(prompt) // 4, len(self.answer) // 4,
                        estimated=True, provider=self.provider)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh keyword-only store installed everywhere the global one is referenced."""
    return install_store(VectorStore(tmp_path / "vs", embedder=None), tmp_path, monkeypatch)


def install_store(new_store, tmp_path, monkeypatch):
    import app.api.documents
    import app.api.status
    import app.api.upload
    import app.generation.answerer
    import app.retrieval.retriever
    import app.retrieval.vectorstore
    for module in (app.retrieval.vectorstore, app.retrieval.retriever, app.generation.answerer,
                   app.api.upload, app.api.documents, app.api.status):
        monkeypatch.setattr(module, "vector_store", new_store)
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(settings, "UPLOAD_DIR", upload_dir)
    return new_store


@pytest.fixture
def use_llm(monkeypatch):
    """Installs a FakeLLM as the active LLM client."""
    import app.generation.llm as llm_module
    import app.retrieval.query_rewriter as rewriter

    def _install(fake):
        monkeypatch.setattr(llm_module, "llm", fake)
        monkeypatch.setattr(rewriter, "llm", fake)
        return fake
    return _install


@pytest.fixture
def make_doc():
    def _make(doc_id="doc1", filename="report.txt", text="", pages=None):
        return {
            "doc_id": doc_id,
            "filename": filename,
            "pages": pages or [{"page": 1, "text": text}],
        }
    return _make
