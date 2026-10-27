import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from app.core.config import settings
from app.retrieval.chunker import chunk_search_text
from app.retrieval.embedder import Embedder, EmbeddingError, create_embedder
from app.retrieval.keyword import BM25Index
from app.utils.logging import logger

_UNSET = object()
FORMAT_VERSION = 2
# After a failed index rebuild, wait this long before trying again (avoids hammering a failing API).
REINDEX_RETRY_SECONDS = 300


def _atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class VectorStore:
    """
    Chunk store with a FAISS cosine-similarity index and a BM25 keyword index.

    Files (in VECTOR_STORE_DIR):
      store.json  - documents registry, chunks, and the embedding signature
      index.faiss - one vector per chunk, in the same order as `chunks`

    Invariant: the FAISS index is either fully in sync with `chunks` and built by the
    current embedder, or unavailable (keyword-only retrieval until `ensure_index` succeeds).
    """

    def __init__(self, store_dir: Optional[Path] = None, embedder=_UNSET):
        self.save_dir = Path(store_dir or settings.VECTOR_STORE_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.save_dir / "store.json"
        self.index_path = self.save_dir / "index.faiss"
        self.embedder: Optional[Embedder] = create_embedder() if embedder is _UNSET else embedder

        self._lock = threading.RLock()
        self.chunks: List[Dict[str, Any]] = []
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.index: Optional[faiss.Index] = None
        self.index_signature: Optional[str] = None
        self.last_index_error: Optional[str] = None
        self._last_failed_attempt = 0.0
        self._bm25: Optional[BM25Index] = None
        self._load()

    # ------------------------------------------------------------------ properties
    @property
    def embedding_signature(self) -> str:
        return self.embedder.signature if self.embedder else "none"

    @property
    def vectors_ready(self) -> bool:
        return (
            self.embedder is not None
            and self.index is not None
            and self.index.ntotal == len(self.chunks)
            and self.index_signature == self.embedding_signature
        )

    # ------------------------------------------------------------------ persistence
    def _load(self):
        legacy_meta = self.save_dir / "metadata.json"
        try:
            if self.store_path.exists():
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                self.chunks = data.get("chunks", [])
                self.documents = data.get("documents", {})
                self.index_signature = data.get("embedding_signature")
                if self.index_path.exists() and self.index_signature == self.embedding_signature:
                    self.index = faiss.read_index(str(self.index_path))
            elif legacy_meta.exists():
                # Pre-v2 format: a bare list of chunks. Its vectors may come from mixed
                # models, so they are discarded and rebuilt by ensure_index().
                logger.info("Migrating legacy vector store format; chunks will be re-embedded.")
                self.chunks = json.loads(legacy_meta.read_text(encoding="utf-8"))
                self.documents = self._documents_from_chunks(self.chunks)
                self._save()
        except Exception as e:
            logger.error(f"Could not load vector store ({e}). Starting with an empty store.")
            self.chunks, self.documents, self.index = [], {}, None

        if self.index is not None and self.index.ntotal != len(self.chunks):
            logger.warning("Vector index is out of sync with stored chunks; it will be rebuilt.")
            self.index = None
        logger.info(f"Vector store loaded: {len(self.documents)} documents, {len(self.chunks)} chunks, "
                    f"vectors {'ready' if self.vectors_ready else 'pending'} ({self.embedding_signature}).")

    @staticmethod
    def _documents_from_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        docs: Dict[str, Dict[str, Any]] = {}
        for c in chunks:
            d = docs.setdefault(c["doc_id"], {
                "doc_id": c["doc_id"],
                "filename": c.get("filename", "Unknown"),
                "file_type": (c.get("filename", "").rsplit(".", 1)[-1] or "txt").lower(),
                "size_bytes": 0,
                "page_count": 1,
                "chunk_count": 0,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            d["chunk_count"] += 1
            d["page_count"] = max(d["page_count"], c.get("page_end") or c.get("page") or 1)
        return docs

    def _save(self):
        data = {
            "version": FORMAT_VERSION,
            "embedding_signature": self.index_signature if self.index is not None else None,
            "documents": self.documents,
            "chunks": self.chunks,
        }
        _atomic_write_text(self.store_path, json.dumps(data, ensure_ascii=False))
        if self.index is not None:
            tmp = self.index_path.with_suffix(".faiss.tmp")
            faiss.write_index(self.index, str(tmp))
            os.replace(tmp, self.index_path)
        elif self.index_path.exists():
            self.index_path.unlink()

    # ------------------------------------------------------------------ indexing
    def _embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        return self.embedder.embed_documents([chunk_search_text(c) for c in chunks])

    def _build_index(self, vectors: np.ndarray) -> faiss.Index:
        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on normalized vectors = cosine
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return index

    def ensure_index(self, force: bool = False) -> bool:
        """(Re)builds the vector index when it is missing or was built by another embedding model."""
        with self._lock:
            if self.vectors_ready or not self.embedder:
                return self.vectors_ready
            if not force and time.monotonic() - self._last_failed_attempt < REINDEX_RETRY_SECONDS:
                return False
            if not self.chunks:
                self.index, self.index_signature = None, self.embedding_signature
                return True
            logger.info(f"Embedding {len(self.chunks)} chunks with {self.embedder.description}...")
            try:
                vectors = self._embed_chunks(self.chunks)
            except EmbeddingError as e:
                self.last_index_error = str(e)
                self._last_failed_attempt = time.monotonic()
                logger.error(f"Vector index rebuild failed: {e}. Using keyword-only retrieval for now.")
                return False
            self.index = self._build_index(vectors)
            self.index_signature = self.embedding_signature
            self.last_index_error = None
            self._save()
            logger.info("Vector index rebuilt.")
            return True

    def add_document(self, doc_info: Dict[str, Any], chunks: List[Dict[str, Any]]):
        """
        Adds a parsed document and its chunks. Embedding happens before anything is
        mutated, so a failure (e.g. API quota) leaves the store unchanged.
        """
        if not chunks:
            raise ValueError("Document produced no chunks.")
        if self.embedder and not self.vectors_ready and self.chunks and not self.ensure_index():
            raise EmbeddingError(self.last_index_error or "Vector index is unavailable.")

        # Embed outside the lock: this can take a while and must not block searches.
        vectors = self._embed_chunks(chunks) if self.embedder else None

        with self._lock:
            if vectors is not None:
                if self.index is None:
                    self.index = self._build_index(vectors)
                elif vectors.shape[1] != self.index.d:
                    raise EmbeddingError(
                        f"Embedding dimension changed ({vectors.shape[1]} vs {self.index.d}); restart to re-index."
                    )
                else:
                    self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
                self.index_signature = self.embedding_signature

            self.chunks.extend(chunks)
            self.documents[doc_info["doc_id"]] = {**doc_info, "chunk_count": len(chunks)}
            self._bm25 = None
            self._save()
            logger.info(f"Indexed {len(chunks)} chunks for '{doc_info.get('filename')}'. Total chunks: {len(self.chunks)}")

    def delete_document(self, doc_id: str) -> bool:
        """Removes a document. Remaining vectors are reused, never re-embedded."""
        with self._lock:
            if doc_id not in self.documents and not any(c["doc_id"] == doc_id for c in self.chunks):
                return False
            keep = np.array([c["doc_id"] != doc_id for c in self.chunks], dtype=bool)
            if self.index is not None and self.index.ntotal == len(self.chunks):
                remaining = self.index.reconstruct_n(0, self.index.ntotal)[keep] if len(keep) else None
                self.index = self._build_index(remaining) if remaining is not None and len(remaining) else None
            else:
                self.index = None
            self.chunks = [c for c, k in zip(self.chunks, keep) if k]
            info = self.documents.pop(doc_id, None)
            self._bm25 = None
            self._save()

        stored = (info or {}).get("stored_name")
        if stored:
            try:
                (settings.UPLOAD_DIR / stored).unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"Could not delete stored file {stored}: {e}")
        logger.info(f"Deleted document {doc_id}. Remaining chunks: {len(self.chunks)}")
        return True

    # ------------------------------------------------------------------ search
    def vector_search(self, query_vectors: np.ndarray, top_k: int,
                      doc_filter: Optional[List[str]] = None) -> List[List[Tuple[Dict[str, Any], float]]]:
        """Cosine search for each query vector. Returns one ranked result list per query."""
        with self._lock:
            if not self.vectors_ready or self.index.ntotal == 0 or query_vectors is None or not len(query_vectors):
                return [[] for _ in range(0 if query_vectors is None else len(query_vectors))]
            allowed = set(doc_filter) if doc_filter else None
            fetch_k = self.index.ntotal if allowed else min(top_k, self.index.ntotal)
            scores, indices = self.index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), fetch_k)
            results = []
            for row_scores, row_idx in zip(scores, indices):
                row = []
                for score, idx in zip(row_scores, row_idx):
                    if idx < 0:
                        continue
                    chunk = self.chunks[idx]
                    if allowed and chunk["doc_id"] not in allowed:
                        continue
                    row.append((chunk, float(score)))
                    if len(row) >= top_k:
                        break
                results.append(row)
            return results

    def keyword_search(self, query: str, top_k: int,
                       doc_filter: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], float]]:
        with self._lock:
            if not self.chunks:
                return []
            if self._bm25 is None:
                self._bm25 = BM25Index([chunk_search_text(c) for c in self.chunks])
            allowed = set(doc_filter) if doc_filter else None
            scored = [
                (self.chunks[idx], score) for idx, score in self._bm25.scores(query).items()
                if score > 0 and (not allowed or self.chunks[idx]["doc_id"] in allowed)
            ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------ accessors
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return list(self.chunks)

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        return [c for c in self.chunks if c["doc_id"] == doc_id]

    def get_documents(self) -> List[Dict[str, Any]]:
        return sorted(self.documents.values(), key=lambda d: d.get("created_at", ""), reverse=True)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.get(doc_id)

    def find_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        return next((d for d in self.documents.values() if d.get("filename") == filename), None)


vector_store = VectorStore()
