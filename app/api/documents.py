from fastapi import APIRouter, HTTPException
from app.core.schemas import DocumentListResponse, DocumentInfo, SourceDetail
from app.retrieval.vectorstore import vector_store

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents", response_model=DocumentListResponse)
def list_documents():
    """Lists indexed documents, newest first."""
    documents = [DocumentInfo(**d) for d in vector_store.get_documents()]
    return DocumentListResponse(
        documents=documents,
        total_count=len(documents),
        total_chunks=sum(d.chunk_count for d in documents),
    )


@router.get("/documents/{doc_id}", response_model=SourceDetail)
@router.get("/sources/{doc_id}", response_model=SourceDetail, include_in_schema=False)
def get_source_detail(doc_id: str):
    """Returns a document's extracted text and chunks (for inspecting what was indexed)."""
    info = vector_store.get_document(doc_id)
    chunks = vector_store.get_document_chunks(doc_id)
    if not info or not chunks:
        raise HTTPException(status_code=404, detail="Document not found.")
    return SourceDetail(
        doc_id=doc_id,
        filename=info["filename"],
        file_type=info.get("file_type", "txt"),
        raw_text="\n\n".join(c.get("text", "") for c in chunks),
        chunk_count=len(chunks),
        chunks=chunks,
    )


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Deletes a document, its vectors and its stored file."""
    if not vector_store.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"success": True, "message": f"Deleted document {doc_id}"}
