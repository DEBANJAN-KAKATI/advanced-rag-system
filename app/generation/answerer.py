import uuid
import time
import re
import json
from typing import List, Dict, Any, Optional
from collections import OrderedDict
from app.core.config import settings
from app.core.schemas import AnswerStyle, ChatResponse, Citation
from app.retrieval.retriever import retrieve_relevant_chunks
from app.generation.memory import chat_memory
from app.generation.prompts import SYSTEM_RAG_PROMPT_TEMPLATE, STYLE_INSTRUCTIONS, FOLLOWUP_QUESTIONS_PROMPT
from app.generation.citation import build_citations, evaluate_confidence
from app.utils.logging import logger

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Maximum number of retries when rate-limited
MAX_RETRIES = 3
BASE_RETRY_DELAY = 5  # seconds


def _parse_retry_delay(error_message: str) -> float:
    """Extract the retry delay from a 429 error message, or return a default."""
    match = re.search(r'retryDelay.*?(\d+(?:\.\d+)?)\s*s', str(error_message))
    if match:
        return min(float(match.group(1)) + 1, 60)  # cap at 60s, add 1s buffer
    return BASE_RETRY_DELAY


def _call_gemini_with_retry(client, model: str, contents: str) -> str:
    """Call Gemini API with automatic retry on rate-limit (429) errors."""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            error_str = str(e)
            # Only retry on rate-limit (429) errors
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < MAX_RETRIES:
                    delay = _parse_retry_delay(error_str)
                    logger.info(f"Rate limited (attempt {attempt+1}/{MAX_RETRIES+1}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
            # Non-retryable error — break immediately
            logger.warning(f"GenAI generation failed: {e}.")
            break
    
    logger.warning(f"GenAI generation failed after {MAX_RETRIES+1} attempts: {last_error}.")
    return ""


def _generate_followup_questions(client, question: str, answer_text: str) -> List[str]:
    """Generate AI-powered follow-up question suggestions using Gemini."""
    try:
        # Truncate answer to first 500 chars to save tokens
        answer_snippet = answer_text[:500]
        prompt = FOLLOWUP_QUESTIONS_PROMPT.format(
            question=question,
            answer_snippet=answer_snippet
        )
        raw = _call_gemini_with_retry(client, settings.LLM_MODEL, prompt)
        if not raw:
            return []
        # Clean markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if isinstance(q, str) and q.strip()][:4]
    except Exception as e:
        logger.warning(f"Follow-up question generation failed: {e}")
    return []


def format_context_passages(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into clear, cited text blocks for prompt context."""
    if not chunks:
        return "No documents available in context."
        
    passages = []
    for idx, chunk in enumerate(chunks):
        fn = chunk.get("filename", "Unknown")
        pg = chunk.get("page", 1)
        sec = chunk.get("section", "General")
        txt = chunk.get("text", "")
        passages.append(
            f"[Source {idx+1}] File: {fn} | Page: {pg} | Section: {sec}\nContent:\n{txt}"
        )
    return "\n\n".join(passages)


def _build_offline_fallback(question: str, chunks: List[Dict[str, Any]], style: AnswerStyle) -> str:
    """
    Build a comprehensive offline answer when the Gemini API is unavailable.
    Respects the answer style mode:
    - CONCISE: Show only the single most relevant passage, briefly
    - MEDIUM: Show top 2-3 passages with page references
    - LONG: Show ALL passages grouped by document & page, full text
    """
    grouped = OrderedDict()
    for chunk in chunks:
        fn = chunk.get("filename", "Unknown")
        pg = chunk.get("page", 1)
        sec = chunk.get("section", "General")
        txt = chunk.get("text", "").strip()
        key = (fn, pg)
        if key not in grouped:
            grouped[key] = {"filename": fn, "page": pg, "sections": [], "texts": []}
        if sec and sec not in grouped[key]["sections"]:
            grouped[key]["sections"].append(sec)
        if txt:
            grouped[key]["texts"].append(txt)

    parts = []

    if style == AnswerStyle.CONCISE:
        # Short & direct — only the top passage
        parts.append("**Quick Answer from Documents:**\n")
        top_chunk = chunks[0] if chunks else None
        if top_chunk:
            txt = top_chunk.get("text", "").strip()
            # Take only the first 2-3 sentences
            sentences = re.split(r'(?<=[.!?])\s+', txt)
            short_text = " ".join(sentences[:3])
            fn = top_chunk.get("filename", "Unknown")
            pg = top_chunk.get("page", 1)
            parts.append(f"{short_text}")
            parts.append(f"\n*[Source: {fn}, Page {pg}]*")
        parts.append("\n\n*⚠️ Offline mode — Gemini API unavailable. Showing direct document extract.*")

    elif style == AnswerStyle.MEDIUM:
        # Exam-style answer — top 2-3 passages, structured
        parts.append(f"### {question}\n")
        items = list(grouped.items())[:3]
        for (fn, pg), data in items:
            section_label = ", ".join(data["sections"]) if data["sections"] else "General"
            combined_text = " ".join(data["texts"])
            # Take first ~300 chars worth of content per passage
            sentences = re.split(r'(?<=[.!?])\s+', combined_text)
            medium_text = " ".join(sentences[:5])
            parts.append(f"**{data['filename']}** (Page {data['page']}, {section_label}):")
            parts.append(f"{medium_text}\n")

        parts.append("\n*⚠️ Offline mode — Gemini API unavailable. Showing key document passages.*")

    else:
        # DETAILED — everything, grouped by page
        parts.append(f"### Detailed Information from Your Documents\n")
        parts.append(f"**Your Question:** *{question}*\n")
        parts.append(f"Found **{len(chunks)} relevant passages** across **{len(grouped)} page(s)**. "
                     f"Below is all the related content extracted from your documents:\n")
        parts.append("---\n")

        for (fn, pg), data in grouped.items():
            section_label = ", ".join(data["sections"]) if data["sections"] else "General"
            parts.append(f"#### 📄 {data['filename']} — Page {data['page']}")
            parts.append(f"**Section:** {section_label}\n")
            
            # Output the FULL text of every chunk from this page (no truncation)
            for text_block in data["texts"]:
                parts.append(f"{text_block}\n")
            
            parts.append("---\n")

        parts.append("*⚠️ Offline mode — This answer was compiled directly from your document passages because the "
                     "Gemini API is temporarily unavailable (rate limit reached). "
                     "The AI-synthesized answer will be available once the API quota resets.*")

    return "\n".join(parts)


def _generate_offline_followups(question: str, chunks: List[Dict[str, Any]]) -> List[str]:
    """Generate basic follow-up question suggestions without AI, using chunk metadata."""
    suggestions = []
    topics_seen = set()
    
    for chunk in chunks:
        fn = chunk.get("filename", "Unknown")
        sec = chunk.get("section", "General")
        txt = chunk.get("text", "")
        
        # Extract potential topic keywords from section names
        if sec and sec != "General" and sec.lower() not in topics_seen:
            topics_seen.add(sec.lower())
            suggestions.append(f"Tell me more about {sec}")
        
    # Add generic follow-ups based on the question
    if "what" in question.lower() or "define" in question.lower():
        suggestions.append(f"Can you show an example of this?")
        suggestions.append(f"What are the key formulas involved?")
    elif "how" in question.lower():
        suggestions.append(f"What are the advantages and disadvantages?")
        suggestions.append(f"Show me a numerical example")
    elif "explain" in question.lower():
        suggestions.append(f"What are the types or classifications?")
        suggestions.append(f"How is this calculated in practice?")
    else:
        suggestions.append(f"Explain this with an example")
        suggestions.append(f"What are the key points to remember?")
    
    suggestions.append("Summarize the entire document")
    
    return suggestions[:4]


def generate_answer(
    question: str,
    style: AnswerStyle = AnswerStyle.MEDIUM,
    document_ids: Optional[List[str]] = None,
    conversation_id: Optional[str] = None
) -> ChatResponse:
    """
    Executes complete RAG answer generation pipeline.
    """
    if not conversation_id:
        conversation_id = str(uuid.uuid4())[:8]

    # 1. Retrieve chunks
    scored_chunks, query_expansions = retrieve_relevant_chunks(
        question=question,
        top_k=settings.RERANK_TOP_K,
        doc_filter=document_ids
    )

    chunks = [c for c, _ in scored_chunks]
    context_text = format_context_passages(chunks)

    # 2. Format memory & prompt
    history_text = chat_memory.format_history_for_prompt(conversation_id)
    style_instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS[AnswerStyle.MEDIUM])

    full_prompt = SYSTEM_RAG_PROMPT_TEMPLATE.format(
        style_instruction=style_instruction,
        chat_history=history_text,
        context_passages=context_text,
        question=question
    )

    # 3. Call Gemini LLM with retry
    answer_text = ""
    gemini_available = False
    client = None
    if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your_api_key":
        if GENAI_AVAILABLE:
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            answer_text = _call_gemini_with_retry(client, settings.LLM_MODEL, full_prompt)
            if answer_text:
                gemini_available = True

    if not answer_text:
        # Offline fallback: style-aware document compilation
        if chunks:
            answer_text = _build_offline_fallback(question, chunks, style)
        else:
            answer_text = "Based on the uploaded documents, there is insufficient evidence to answer this question."

    # 4. Generate follow-up questions
    suggested_questions = []
    if gemini_available and client:
        suggested_questions = _generate_followup_questions(client, question, answer_text)
    
    # If Gemini failed or returned nothing, use offline follow-ups
    if not suggested_questions and chunks:
        suggested_questions = _generate_offline_followups(question, chunks)

    # 5. Build Citations & Evaluate Confidence
    citations = build_citations(scored_chunks)
    confidence = evaluate_confidence(answer_text, scored_chunks)

    # 6. Record to memory
    chat_memory.add_turn(conversation_id, question, answer_text)

    sources_used = [
        {
            "doc_id": c.get("doc_id"),
            "filename": c.get("filename"),
            "page": c.get("page"),
            "section": c.get("section")
        } for c in chunks
    ]

    return ChatResponse(
        answer=answer_text,
        style=style,
        citations=citations,
        confidence=confidence,
        query_expansions=query_expansions,
        conversation_id=conversation_id,
        sources_used=sources_used,
        suggested_questions=suggested_questions
    )
