import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.config import settings
from app.core.schemas import UploadResponse
from app.ingestion.loader import EmptyDocumentError, compute_doc_id, load_document
from app.retrieval.chunker import chunk_document
from app.retrieval.embedder import EmbeddingError
from app.retrieval.vectorstore import vector_store
from app.utils.filetypes import get_extension, is_supported_file, supported_extensions_label
from app.utils.logging import logger

router = APIRouter(prefix="/api", tags=["upload"])


class UploadTooLarge(ValueError):
    pass


def _save_upload(file: UploadFile, dest: Path):
    """Streams the upload to disk, enforcing MAX_UPLOAD_MB."""
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with open(dest, "wb") as out:
        while block := file.file.read(1 << 20):
            written += len(block)
            if written > limit:
                raise UploadTooLarge(f"File exceeds the {settings.MAX_UPLOAD_MB} MB upload limit.")
            out.write(block)


def _error(filename: str, message: str) -> UploadResponse:
    return UploadResponse(success=False, filename=filename, doc_id="", chunks_created=0,
                          page_count=0, message=message, status="error")


def _ingest_one(file: UploadFile) -> UploadResponse:
    # Never trust the client's path: keep only the base name for display.
    filename = Path((file.filename or "").replace("\\", "/")).name.strip()
    if not filename:
        return _error("(unnamed)", "File has no name.")
    if not is_supported_file(filename):
        return _error(filename, f"Unsupported file format. Supported: {supported_extensions_label()}.")

    ext = get_extension(filename)
    incoming = settings.UPLOAD_DIR / f".incoming-{uuid.uuid4().hex}{ext}"
    stored_path = None
    try:
        _save_upload(file, incoming)
        doc_id = compute_doc_id(incoming)

        existing = vector_store.get_document(doc_id)
        if existing:
            incoming.unlink(missing_ok=True)
            return UploadResponse(success=True, filename=filename, doc_id=doc_id,
                                  chunks_created=0, page_count=existing.get("page_count", 1),
                                  message=f"Already indexed as '{existing['filename']}'.", status="duplicate")

        stored_path = settings.UPLOAD_DIR / f"{doc_id}{ext}"
        os.replace(incoming, stored_path)

        doc = load_document(stored_path, display_name=filename, doc_id=doc_id)
        chunks = chunk_document(doc)
        previous = vector_store.find_by_filename(filename)

        vector_store.add_document({
            "doc_id": doc_id,
            "filename": filename,
            "file_type": ext.lstrip("."),
            "size_bytes": doc["size_bytes"],
            "page_count": doc.get("total_pages", 1),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stored_name": stored_path.name,
        }, chunks)
        stored_path = None  # now owned by the store

        status, message = "indexed", f"Indexed {len(chunks)} chunks."
        if previous and previous["doc_id"] != doc_id:
            # Uploading a changed file with the same name replaces the old version.
            vector_store.delete_document(previous["doc_id"])
            status, message = "replaced", f"Replaced the previous version; indexed {len(chunks)} chunks."

        return UploadResponse(success=True, filename=filename, doc_id=doc_id, chunks_created=len(chunks),
                              page_count=doc.get("total_pages", 1), message=message, status=status)
    except (UploadTooLarge, EmptyDocumentError, EmbeddingError) as e:
        logger.warning(f"Upload of {filename} rejected: {e}")
        return _error(filename, str(e))
    except Exception as e:
        logger.exception(f"Error processing file {filename}")
        return _error(filename, f"Could not process this file: {e}")
    finally:
        incoming.unlink(missing_ok=True)
        if stored_path is not None:
            stored_path.unlink(missing_ok=True)


@router.post("/upload", response_model=List[UploadResponse])
def upload_files(files: List[UploadFile] = File(...)):
    """
    Accepts one or more documents, parses, chunks and indexes them.
    Each file gets its own result; one bad file does not fail the others.
    (Sync endpoint: FastAPI runs it in a worker thread so the server stays responsive.)
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    return [_ingest_one(f) for f in files]
