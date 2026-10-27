import json
import re
from typing import List, Optional, Tuple
from app.core.config import settings
from app.generation.llm import LLMError, llm
from app.generation.prompts import QUERY_REWRITE_PROMPT, QUERY_REWRITE_SYSTEM
from app.utils.logging import logger


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0) if match else text)


def rewrite_query(question: str, chat_history: str = "", expand: Optional[bool] = None) -> Tuple[str, List[str]]:
    """
    Returns (standalone_question, search_queries).

    With conversation history, follow-ups like "what about Q4?" are rewritten into
    self-contained questions before searching. Alternative phrasings are added when
    `expand` (default: QUERY_REWRITE) is on. Without an LLM, the question is used as-is.
    """
    question = question.strip()
    expand = settings.QUERY_REWRITE if expand is None else expand
    if llm is None or not (expand or chat_history):
        return question, [question]

    try:
        raw = llm.generate(
            QUERY_REWRITE_SYSTEM,
            QUERY_REWRITE_PROMPT.format(chat_history=chat_history, question=question),
            json_mode=True,
            fast=True,
            purpose="query_rewrite",
        )
        parsed = _parse_json_object(raw)
    except (LLMError, ValueError, TypeError) as e:
        logger.warning(f"Query rewriting skipped: {e}")
        return question, [question]

    standalone = parsed.get("standalone") if isinstance(parsed, dict) else None
    standalone = standalone.strip() if isinstance(standalone, str) and standalone.strip() else question

    queries = [standalone]
    if question.lower() != standalone.lower():
        queries.append(question)
    if expand and isinstance(parsed, dict):
        for q in parsed.get("queries") or []:
            if isinstance(q, str) and q.strip() and q.strip().lower() not in {x.lower() for x in queries}:
                queries.append(q.strip())
    return standalone, queries[:4]
