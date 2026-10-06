"""
Embedding backends for the evaluation harness.

`onnx-minilm` is the same all-MiniLM-L6-v2 model the app uses as its local
fallback (via sentence-transformers), run through onnxruntime instead: mean
pooling + L2 normalisation, identical outputs, no PyTorch needed. The ONNX
export is the one chromadb ships; it is downloaded once into data/models/ and
verified against chromadb's pinned SHA-256.

`app` uses whatever the app itself is configured with (Gemini embeddings when a
key is set, otherwise sentence-transformers).
"""
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import List

import numpy as np

from app.core.config import settings
from app.core.telemetry import record_embedding

MINILM_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
MINILM_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
MODEL_DIR = Path(os.getenv("EVAL_MODEL_DIR", settings.DATA_DIR / "models" / "all-MiniLM-L6-v2-onnx"))


def _ensure_minilm(model_dir: Path) -> Path:
    onnx_dir = model_dir / "onnx"
    if (onnx_dir / "model.onnx").exists():
        return onnx_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    archive = model_dir / "onnx.tar.gz"
    print(f"Downloading all-MiniLM-L6-v2 (ONNX, ~80 MB) to {model_dir} ...")
    urllib.request.urlretrieve(MINILM_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != MINILM_SHA256:
        archive.unlink()
        raise RuntimeError(f"Checksum mismatch for {MINILM_URL}: got {digest}")
    with tarfile.open(archive) as tar:
        tar.extractall(model_dir, filter="data")
    archive.unlink()
    return onnx_dir


class OnnxMiniLMEmbedder:
    """Drop-in replacement for app.retrieval.embedder.Embedder."""

    name = "all-MiniLM-L6-v2 (onnx)"
    max_length = 256  # sentence-transformers' max_seq_length for this model

    def __init__(self, model_dir: Path = MODEL_DIR, batch_size: int = 32):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_dir = _ensure_minilm(Path(model_dir))
        self.tokenizer = Tokenizer.from_file(str(onnx_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(onnx_dir / "model.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.batch_size = batch_size

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        out = np.zeros((len(texts), 384), dtype=np.float32)
        # Length-sorted batches keep padding (and CPU time) down.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        n_tokens = 0
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tokenizer.encode_batch([texts[i] for i in idx])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            n_tokens += int(mask.sum())
            hidden = self.session.run(None, {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": np.zeros_like(ids),
            })[0]
            pooled = (hidden * mask[..., None]).sum(axis=1) / np.clip(mask.sum(axis=1, keepdims=True), 1, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out[idx] = pooled
        record_embedding("local-onnx", "all-MiniLM-L6-v2", len(texts), n_tokens, estimated=False)
        return out

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])


def get_embedder(name: str):
    if name == "onnx-minilm":
        return OnnxMiniLMEmbedder()
    if name == "app":
        from app.retrieval.embedder import embedder
        return embedder
    raise ValueError(f"Unknown embedder {name!r}; expected 'onnx-minilm' or 'app'")
