from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from app.core.schemas import DocumentListResponse, DocumentInfo, SourceDetail
from app.retrieval.vectorstore import vector_store
from app.utils.logging import logger

router = APIRouter(prefix="/api", tags=["documents"])

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """Returns list of currently indexed documents with chunk statistics."""
    all_chunks = vector_store.get_all_chunks()
    
    doc_map: Dict[str, Dict[str, Any]] = {}
    for chunk in all_chunks:
        doc_id = chunk.get("doc_id")
        if not doc_id:
            continue
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "doc_id": doc_id,
                "filename": chunk.get("filename", "Unknown"),
                "file_type": chunk.get("filename", "").split(".")[-1].lower(),
                "size_bytes": 0,
                "chunk_count": 0,
                "page_count": 1,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        doc_map[doc_id]["chunk_count"] += 1
        page_num = chunk.get("page", 1)
        if page_num > doc_map[doc_id]["page_count"]:
            doc_map[doc_id]["page_count"] = page_num

    documents = [DocumentInfo(**info) for info in doc_map.values()]
    return DocumentListResponse(
        documents=documents,
        total_count=len(documents),
        total_chunks=len(all_chunks)
    )

@router.get("/sources/{doc_id}", response_model=SourceDetail)
async def get_source_detail(doc_id: str):
    """Retrieves document raw text chunks for source viewer."""
    chunks = vector_store.get_document_chunks(doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found.")

    filename = chunks[0].get("filename", "Unknown")
    file_type = filename.split(".")[-1].lower() if "." in filename else "txt"
    full_text = "\n\n".join([c.get("text", "") for c in chunks])

    return SourceDetail(
        doc_id=doc_id,
        filename=filename,
        file_type=file_type,
        raw_text=full_text,
        chunk_count=len(chunks),
        chunks=chunks
    )

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Deletes document and removes associated vectors from vector store."""
    chunks = vector_store.get_document_chunks(doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    vector_store.delete_document(doc_id)
    return {"success": True, "message": f"Deleted document {doc_id}"}
