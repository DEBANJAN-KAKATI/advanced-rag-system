from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AnswerStyle(str, Enum):
    CONCISE = "concise"
    MEDIUM = "medium"
    LONG = "long"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="User question")
    style: AnswerStyle = Field(default=AnswerStyle.MEDIUM, description="Response length style")
    document_ids: Optional[List[str]] = Field(default=None, description="Restrict the search to these documents")
    conversation_id: Optional[str] = Field(default=None, max_length=64, description="Session ID for chat memory")


class Citation(BaseModel):
    id: str
    number: int = Field(..., description="The [n] marker used in the answer text")
    chunk_id: str
    doc_id: str
    filename: str
    page: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    snippet: str
    text: str = Field("", description="Full passage text")
    score: float = Field(..., description="Relevance score (0-1)")


class ChatResponse(BaseModel):
    answer: str
    style: AnswerStyle
    citations: List[Citation]
    confidence: str  # "High", "Medium", "Low", "Insufficient Evidence"
    query_expansions: List[str]
    standalone_question: Optional[str] = None
    conversation_id: str
    sources_used: List[Dict[str, Any]]
    suggested_questions: List[str] = Field(default_factory=list)
    llm_used: bool = Field(False, description="False when the answer was extracted without an LLM")
    notice: Optional[str] = Field(None, description="Warning to show with the answer (e.g. LLM unavailable)")
    retrieval_mode: str = "hybrid"
    usage: Optional[Dict[str, Any]] = Field(
        default=None, description="Per-query latency (ms) by stage, LLM/embedding token counts and estimated cost")


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    size_bytes: int
    chunk_count: int
    page_count: Optional[int] = 1
    created_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_count: int
    total_chunks: int


class UploadResponse(BaseModel):
    success: bool
    filename: str
    doc_id: str
    chunks_created: int
    page_count: int
    message: str
    status: str = Field("indexed", description="indexed | duplicate | replaced | error")


class SourceDetail(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    raw_text: str
    chunk_count: int
    chunks: List[Dict[str, Any]]


class StatusResponse(BaseModel):
    llm_provider: str
    llm_model: Optional[str]
    llm_available: bool
    embedding_provider: str
    vectors_ready: bool
    reranker: str
    documents: int
    chunks: int
    max_upload_mb: int
    supported_formats: str
    warnings: List[str] = Field(default_factory=list)
