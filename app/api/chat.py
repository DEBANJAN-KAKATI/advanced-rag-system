from fastapi import APIRouter, HTTPException
from app.core.schemas import QueryRequest, ChatResponse
from app.generation.answerer import generate_answer
from app.utils.logging import logger

router = APIRouter(prefix="/api", tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: QueryRequest):
    """
    RAG Chat endpoint. Accepts question, answer style (concise/medium/long),
    optional document filters, and conversation_id.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = generate_answer(
            question=request.question,
            style=request.style,
            document_ids=request.document_ids,
            conversation_id=request.conversation_id
        )
        return response
    except Exception as e:
        logger.error(f"Chat generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")
