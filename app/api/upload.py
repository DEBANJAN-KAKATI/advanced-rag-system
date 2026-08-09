import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.config import settings
from app.core.schemas import UploadResponse
from app.utils.filetypes import is_supported_file
from app.ingestion.loader import load_document
from app.retrieval.chunker import chunk_document
from app.retrieval.vectorstore import vector_store
from app.utils.logging import logger

router = APIRouter(prefix="/api", tags=["upload"])

@router.post("/upload", response_model=List[UploadResponse])
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Accepts one or multiple document files (PDF, DOCX, HTML, TXT),
    saves them to disk, parses text, chunks intelligently, and indexes vectors into FAISS.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    responses = []

    for file in files:
        if not is_supported_file(file.filename):
            logger.warning(f"Unsupported file type submitted: {file.filename}")
            responses.append(UploadResponse(
                success=False,
                filename=file.filename,
                doc_id="",
                chunks_created=0,
                page_count=0,
                message=f"Unsupported file format. Allowed formats: PDF, DOCX, HTML, TXT, MD, CSV, JSON."
            ))
            continue

        try:
            # Save file to upload directory
            save_path = settings.UPLOAD_DIR / file.filename
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Ingest & Load
            doc_data = load_document(save_path)
            doc_id = doc_data["doc_id"]

            # Chunk
            chunks = chunk_document(doc_data)

            # Index into vector store
            vector_store.add_chunks(chunks)

            responses.append(UploadResponse(
                success=True,
                filename=file.filename,
                doc_id=doc_id,
                chunks_created=len(chunks),
                page_count=doc_data.get("total_pages", 1),
                message=f"Successfully uploaded and indexed {len(chunks)} chunks."
            ))
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            responses.append(UploadResponse(
                success=False,
                filename=file.filename,
                doc_id="",
                chunks_created=0,
                page_count=0,
                message=f"Error ingesting document: {str(e)}"
            ))

    return responses
