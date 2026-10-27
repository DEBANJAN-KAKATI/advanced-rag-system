import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.core.schemas import QueryRequest, ChatResponse
from app.generation.answerer import generate_answer, stream_answer
from app.generation.memory import chat_memory
from app.utils.logging import logger

router = APIRouter(prefix="/api", tags=["chat"])


def _validate(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: QueryRequest):
    """Answers a question and returns the complete response as JSON."""
    _validate(request)
    try:
        return generate_answer(
            question=request.question,
            style=request.style,
            document_ids=request.document_ids,
            conversation_id=request.conversation_id,
        )
    except Exception as e:
        logger.exception("Chat generation error")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")


@router.post("/chat/stream")
def chat_stream_endpoint(request: QueryRequest):
    """
    Answers a question as Server-Sent Events:
    `sources` (retrieved passages), `token` (answer text deltas), `done` (final ChatResponse) or `error`.
    """
    _validate(request)

    def events():
        try:
            for event, data in stream_answer(
                question=request.question,
                style=request.style,
                document_ids=request.document_ids,
                conversation_id=request.conversation_id,
            ):
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("Streaming chat error")
            yield f"event: error\ndata: {json.dumps({'message': f'Failed to generate answer: {e}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.delete("/conversations/{conversation_id}")
def clear_conversation(conversation_id: str):
    """Forgets a conversation's history (used by the "New chat" button)."""
    chat_memory.clear(conversation_id)
    return {"success": True}
