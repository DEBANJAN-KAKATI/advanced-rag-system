import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.core.telemetry import stage
from app.retrieval.embedder import embedder as default_embedder
from app.utils.logging import logger

class VectorStore:
    def __init__(self, embedder=None, persist: bool = True, save_dir: Path = None):
        """
        persist=False gives an in-memory store that never touches disk, which the
        evaluation harness uses so experiments don't overwrite the user's index.
        """
        self.embedder = embedder or default_embedder
        self.persist = persist
        self.save_dir = save_dir or settings.VECTOR_STORE_DIR
        self.index_path = self.save_dir / "faiss.index"
        self.meta_path = self.save_dir / "metadata.json"
        
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = None
        if self.persist:
            self._load()

    def _load(self):
        """Loads FAISS index and metadata if exists."""
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                self.dimension = self.index.d
                logger.info(f"Loaded VectorStore with {self.index.ntotal} vectors (dim={self.dimension}).")
            except Exception as e:
                logger.warning(f"Error loading vectorstore: {e}. Initializing fresh store.")
                self.index = None
                self.metadata = []

    def _save(self):
        """Persists index and metadata to disk."""
        if self.persist and self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Embeds and adds document chunks to FAISS index."""
        if not chunks:
            return
            
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        if embeddings.shape[0] == 0:
            return
            
        num_vecs, dim = embeddings.shape
        
        if self.index is None or self.dimension != dim:
            # Inner product for normalized vectors = cosine similarity
            self.index = faiss.IndexFlatIP(dim)
            self.dimension = dim
            self.metadata = []
            
        self.index.add(embeddings)
        self.metadata.extend(chunks)
        self._save()
        logger.info(f"Added {num_vecs} chunks to FAISS vector store. Total: {self.index.ntotal}")

    def similarity_search(self, query: str, top_k: int = 15, doc_filter: List[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Performs vector similarity search."""
        if self.index is None or self.index.ntotal == 0:
            return []
            
        with stage("embed_query"):
            query_vec = self.embedder.embed_query(query)
        # Search extra candidates if doc_filter is specified
        fetch_k = min(top_k * 4 if doc_filter else top_k, self.index.ntotal)
        
        with stage("vector_search"):
            scores, indices = self.index.search(query_vec, fetch_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            if doc_filter and item.get("doc_id") not in doc_filter:
                continue
            results.append((item, float(score)))
            if len(results) >= top_k:
                break
                
        return results

    def delete_document(self, doc_id: str):
        """Removes all chunks associated with doc_id and rebuilds index."""
        if not self.metadata:
            return
            
        keep_metadata = []
        keep_indices = []
        
        for idx, item in enumerate(self.metadata):
            if item.get("doc_id") != doc_id:
                keep_metadata.append(item)
                keep_indices.append(idx)
                
        if len(keep_metadata) == len(self.metadata):
            return  # Nothing to delete
            
        if not keep_metadata:
            self.index = None
            self.metadata = []
            self.dimension = None
            if self.index_path.exists():
                os.remove(self.index_path)
            if self.meta_path.exists():
                os.remove(self.meta_path)
            logger.info(f"Deleted doc_id {doc_id}. VectorStore is now empty.")
            return

        # Re-embed remaining chunks to guarantee clean index state
        logger.info(f"Re-indexing after deleting doc_id {doc_id}...")
        texts = [c["text"] for c in keep_metadata]
        embeddings = self.embedder.embed_texts(texts)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.dimension = dim
        self.index.add(embeddings)
        self.metadata = keep_metadata
        self._save()

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return self.metadata

    def get_vectors(self, chunk_ids: List[str]) -> np.ndarray:
        """Returns the stored (normalized) vectors for the given chunk IDs."""
        row_by_id = {c["chunk_id"]: i for i, c in enumerate(self.metadata)}
        return np.vstack([self.index.reconstruct(row_by_id[cid]) for cid in chunk_ids])

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        return [c for c in self.metadata if c.get("doc_id") == doc_id]

vector_store = VectorStore()
