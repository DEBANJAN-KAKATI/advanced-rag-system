import json
from typing import List
from app.core.config import settings
from app.core.telemetry import record_llm_response
from app.utils.logging import logger

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

def rewrite_query(query: str) -> List[str]:
    """
    Generates 2-3 query variations/paraphrases to improve recall in hybrid search.
    Returns original query + expanded queries.
    """
    queries = [query]
    if not settings.has_llm_key or not GENAI_AVAILABLE:
        return queries

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        prompt = f"""You are a search query expansion assistant for a Document RAG system.
Given the user question below, generate 2 alternate search queries or sub-queries that express the same intent or highlight key concepts.

Question: "{query}"

Output ONLY a JSON array of strings, e.g. ["query 1", "query 2"]. Do not add markdown backticks or commentary."""

        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
        )
        text = response.text.strip()
        record_llm_response("query_rewrite", settings.LLM_MODEL, response, prompt, text)
        if text.startswith("```"):
            text = text.split("```")[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
                
        parsed = json.loads(text)
        if isinstance(parsed, list):
            for q in parsed:
                if isinstance(q, str) and q.strip() and q.strip() not in queries:
                    queries.append(q.strip())
    except Exception as e:
        logger.warning(f"Query rewriter encountered issue: {e}. Using original query.")

    return queries[:3]
