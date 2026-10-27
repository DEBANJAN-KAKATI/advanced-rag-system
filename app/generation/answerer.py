import json
import re
import uuid
from typing import List, Dict, Any, Optional, Iterator, Tuple
from app.core.config import settings
from app.core.schemas import AnswerStyle, ChatResponse
from app.core.telemetry import QueryTrace, current_trace, stage, use_trace
from app.generation import llm as llm_module
from app.generation.citation import (
    build_citations, evaluate_confidence, is_not_found_answer, normalize_citation_markers, select_cited,
)
from app.generation.llm import LLMError
from app.generation.memory import chat_memory
from app.generation.prompts import (
    ANSWER_PROMPT_TEMPLATE, FOLLOWUP_INSTRUCTION, FOLLOWUP_MARKER, NOT_FOUND_SENTENCE, STYLE_INSTRUCTIONS, SYSTEM_PROMPT,
)
from app.retrieval.chunker import split_sentences
from app.retrieval.keyword import term_coverage
from app.retrieval.retriever import RetrievalConfig, RetrievalResult, retrieve_relevant_chunks
from app.retrieval.vectorstore import vector_store
from app.utils.logging import logger

Event = Tuple[str, Dict[str, Any]]


# ---------------------------------------------------------------------------- formatting
def _page_label(chunk: Dict[str, Any]) -> str:
    start, end = chunk.get("page"), chunk.get("page_end")
    if start is None:
        return ""
    return f"p. {start}" if not end or end == start else f"pp. {start}-{end}"


def format_context_passages(chunks: List[Dict[str, Any]]) -> str:
    """Numbered passages; the numbers are what the model cites as [n]."""
    passages = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = [chunk.get("filename", "Unknown")]
        if chunk.get("filename", "").lower().endswith(".pdf"):  # other formats have no real pages
            meta.append(_page_label(chunk))
        if chunk.get("section") and chunk["section"] != "General":
            meta.append(f"section: {chunk['section']}")
        passages.append(f"[{idx}] ({' | '.join(meta)})\n{chunk.get('text', '').strip()}")
    return "\n\n".join(passages)


class _MarkerSplitter:
    """Streams answer text while holding back the follow-up marker and everything after it."""

    def __init__(self, marker: str):
        self.marker = marker
        self.buffer = ""
        self.tail = ""
        self.found = False

    def feed(self, text: str) -> str:
        if self.found:
            self.tail += text
            return ""
        self.buffer += text
        idx = self.buffer.find(self.marker)
        if idx >= 0:
            emit, self.tail, self.buffer, self.found = self.buffer[:idx], self.buffer[idx + len(self.marker):], "", True
            return emit
        keep = next((k for k in range(min(len(self.marker) - 1, len(self.buffer)), 0, -1)
                     if self.buffer.endswith(self.marker[:k])), 0)
        emit, self.buffer = self.buffer[:len(self.buffer) - keep], self.buffer[len(self.buffer) - keep:]
        return emit

    def flush(self) -> str:
        emit, self.buffer = self.buffer, ""
        return emit


def _parse_followups(tail: str) -> List[str]:
    match = re.search(r"\[.*\]", tail or "", re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return []
    return [q.strip() for q in items if isinstance(q, str) and q.strip()][:4] if isinstance(items, list) else []


# ---------------------------------------------------------------------------- no-LLM fallback
def build_extractive_answer(question: str, chunks: List[Dict[str, Any]], style: AnswerStyle, reason: str) -> str:
    """
    Answer assembled from the most relevant sentences of the retrieved passages, used when
    no LLM is configured or the LLM call failed. Each sentence keeps its [n] source marker.
    """
    limit = {AnswerStyle.CONCISE: 2, AnswerStyle.MEDIUM: 5, AnswerStyle.LONG: 10}.get(style, 5)
    candidates = []
    for n, chunk in enumerate(chunks, start=1):
        for pos, (sentence, _) in enumerate(split_sentences(chunk.get("text", ""))):
            if len(sentence) < 25 or sentence.startswith("#"):
                continue
            score = term_coverage(question, sentence) + 0.15 / n
            candidates.append((score, n, pos, sentence))
    best = sorted(candidates, key=lambda x: x[0], reverse=True)[:limit]
    best.sort(key=lambda x: (x[1], x[2]))  # restore reading order

    note = (f"> **Extractive answer** - the AI model is unavailable ({reason}). "
            f"Below are the most relevant sentences from your documents.\n\n")
    if not best:
        return note + NOT_FOUND_SENTENCE
    if style == AnswerStyle.CONCISE:
        return note + " ".join(f"{s} [{n}]" for _, n, _, s in best)
    return note + "\n".join(f"- {s} [{n}]" for _, n, _, s in best)


def _fallback_followups(chunks: List[Dict[str, Any]]) -> List[str]:
    suggestions: List[str] = []
    for chunk in chunks:
        section = (chunk.get("section") or "").split(" > ")[-1]
        if section and section != "General" and len(section) < 60:
            suggestion = f"Tell me more about {section}"
            if suggestion not in suggestions:
                suggestions.append(suggestion)
    filenames = list(dict.fromkeys(c.get("filename") for c in chunks if c.get("filename")))
    if filenames:
        suggestions.append(f"Summarize {filenames[0]}")
    return suggestions[:3]


# ---------------------------------------------------------------------------- pipeline
def _traced(iterator, t: QueryTrace):
    """Advances `iterator` with `t` active for each step (see telemetry.use_trace)."""
    iterator = iter(iterator)
    while True:
        with use_trace(t):
            try:
                item = next(iterator)
            except StopIteration:
                return
        yield item


def stream_answer(
    question: str,
    style: AnswerStyle = AnswerStyle.MEDIUM,
    document_ids: Optional[List[str]] = None,
    conversation_id: Optional[str] = None,
    store=None,
    retrieval_config: Optional[RetrievalConfig] = None,
    generate_followups: bool = True,
) -> Iterator[Event]:
    """
    Full RAG pipeline as a stream of events:
      ("sources", {...})  retrieved passages (numbered like the [n] markers), before generation
      ("token", {"text"}) answer text as it is generated
      ("done", ChatResponse dict) final answer, cited sources, confidence, follow-ups, and
                          `usage`: per-stage latency, token counts and estimated cost

    `store` and `retrieval_config` override the global index and retrieval settings
    (used by the evaluation harness).
    """
    # Join a trace opened by the caller (e.g. the evaluation harness), else own one.
    outer = current_trace()
    t = outer or QueryTrace()
    yield from _traced(_answer_events(question, style, document_ids, conversation_id, store,
                                      retrieval_config, generate_followups, t, owns_trace=outer is None), t)


def _answer_events(question, style, document_ids, conversation_id, store, retrieval_config,
                   generate_followups, t: QueryTrace, owns_trace: bool) -> Iterator[Event]:
    store = store or vector_store
    question = question.strip()
    conversation_id = conversation_id or uuid.uuid4().hex[:12]
    history_text = chat_memory.format_history_for_prompt(conversation_id)
    client = llm_module.llm
    suggest = generate_followups and settings.SUGGEST_FOLLOWUPS

    def finish(answer: str, retrieval: Optional[RetrievalResult], citations, llm_used: bool,
               notice: Optional[str], followups: List[str]) -> Event:
        scored = retrieval.chunks if retrieval else []
        confidence = evaluate_confidence(answer, scored, retrieval.scorer if retrieval else "keyword",
                                         retrieval.low_relevance if retrieval else False,
                                         retrieval.mode if retrieval else "hybrid")
        chat_memory.add_turn(conversation_id, question, answer)
        if owns_trace:
            t.finish()
        t.annotations.setdefault("retrieval_mode", retrieval.mode if retrieval else "none")
        response = ChatResponse(
            answer=answer,
            style=style,
            citations=citations,
            confidence=confidence,
            query_expansions=retrieval.queries if retrieval else [question],
            standalone_question=retrieval.standalone_question if retrieval else question,
            conversation_id=conversation_id,
            sources_used=[{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "filename": c.filename,
                           "page": c.page, "section": c.section} for c in citations],
            suggested_questions=followups if generate_followups else [],
            llm_used=llm_used,
            notice=notice,
            retrieval_mode=retrieval.mode if retrieval else "none",
            usage=t.summary(),
        )
        return "done", response.model_dump(mode="json")

    # 1. Nothing to search
    available = store.get_documents()
    if document_ids:
        available = [d for d in available if d["doc_id"] in set(document_ids)]
    if not available:
        msg = ("No documents are uploaded yet. Upload a PDF, Word, HTML or text file and ask again."
               if not store.get_documents() else "None of the selected documents exist anymore. Please reselect.")
        yield "token", {"text": msg}
        yield finish(msg, None, [], False, None, [])
        return

    # 2. Retrieve
    with stage("retrieval"):
        retrieval = retrieve_relevant_chunks(question, style=style, doc_filter=document_ids,
                                             chat_history=history_text, store=store, config=retrieval_config)
    chunks = [c for c, _ in retrieval.chunks]
    all_citations = build_citations(retrieval.chunks)
    yield "sources", {
        "conversation_id": conversation_id,
        "standalone_question": retrieval.standalone_question,
        "query_expansions": retrieval.queries,
        "citations": [c.model_dump(mode="json") for c in all_citations],
    }
    notice = " ".join(retrieval.notices) or None

    if not chunks:
        yield "token", {"text": NOT_FOUND_SENTENCE}
        yield finish(NOT_FOUND_SENTENCE, retrieval, [], False, notice, [])
        return

    # 3. Generate
    if client is None:
        answer = build_extractive_answer(retrieval.standalone_question, chunks, style, "no LLM is configured")
        yield "token", {"text": answer}
        yield finish(answer, retrieval, select_cited(all_citations, answer), False,
                     notice, _fallback_followups(chunks))
        return

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        chat_history=history_text,
        context_passages=format_context_passages(chunks),
        style_instruction=STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS[AnswerStyle.MEDIUM]),
        question=retrieval.standalone_question if retrieval.standalone_question != question and history_text else question,
        followup_instruction=FOLLOWUP_INSTRUCTION if suggest else "",
    )
    if retrieval.mode == "summary":
        prompt += "\n\nThe sources are excerpts spread across the whole document(s); summarize them as a whole."

    splitter = _MarkerSplitter(FOLLOWUP_MARKER)
    streamed: List[str] = []
    try:
        # Time only the token production, not the time the consumer spends handling events.
        for piece in _timed(client.stream(SYSTEM_PROMPT, prompt, purpose="answer"), "generation"):
            emit = splitter.feed(piece)
            if emit:
                streamed.append(emit)
                yield "token", {"text": emit}
        rest = splitter.flush()
        if rest:
            streamed.append(rest)
            yield "token", {"text": rest}
    except LLMError as e:
        logger.warning(f"LLM generation failed: {e}")
        if not "".join(streamed).strip():
            answer = build_extractive_answer(retrieval.standalone_question, chunks, style, str(e))
            yield "token", {"text": answer}
            yield finish(answer, retrieval, select_cited(all_citations, answer), False,
                         notice, _fallback_followups(chunks))
            return
        streamed.append(f"\n\n*(The answer was cut off: {e}.)*")

    answer = normalize_citation_markers("".join(streamed).strip(), len(chunks))
    if not answer:
        answer = NOT_FOUND_SENTENCE
    followups = _parse_followups(splitter.tail) or ([] if is_not_found_answer(answer) else _fallback_followups(chunks))
    yield finish(answer, retrieval, select_cited(all_citations, answer), True, notice, followups)


def _timed(iterator, name: str):
    """Adds the time spent producing each item of `iterator` to stage `name`."""
    iterator = iter(iterator)
    while True:
        with stage(name):
            try:
                item = next(iterator)
            except StopIteration:
                return
        yield item


def generate_answer(
    question: str,
    style: AnswerStyle = AnswerStyle.MEDIUM,
    document_ids: Optional[List[str]] = None,
    conversation_id: Optional[str] = None,
    store=None,
    retrieval_config: Optional[RetrievalConfig] = None,
    generate_followups: bool = True,
) -> ChatResponse:
    """Non-streaming wrapper around `stream_answer`; the response's `usage` reports cost and latency."""
    final = None
    for event, data in stream_answer(question, style, document_ids, conversation_id,
                                     store, retrieval_config, generate_followups):
        if event == "done":
            final = data
    return ChatResponse(**final)
