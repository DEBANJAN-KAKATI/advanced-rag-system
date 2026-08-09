from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class AnswerStyle(str, Enum):
    CONCISE = "concise"
    MEDIUM = "medium"
    LONG = "long"

class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    style: AnswerStyle = Field(default=AnswerStyle.MEDIUM, description="Response length style")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document filter list")
    conversation_id: Optional[str] = Field(default=None, description="Session ID for chat memory")

class Citation(BaseModel):
    id: str
    doc_id: str
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    style: AnswerStyle
    citations: List[Citation]
    confidence: str  # "High", "Medium", "Low", "Insufficient Evidence"
    query_expansions: List[str]
    conversation_id: str
    sources_used: List[Dict[str, Any]]
    suggested_questions: List[str] = Field(default_factory=list, description="AI-generated follow-up questions")

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

class SourceDetail(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    raw_text: str
    chunk_count: int
    chunks: List[Dict[str, Any]]
