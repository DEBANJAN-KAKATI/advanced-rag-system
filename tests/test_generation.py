from app.core.schemas import AnswerStyle
from app.generation.answerer import _MarkerSplitter, build_extractive_answer, stream_answer
from app.generation.citation import is_not_found_answer, normalize_citation_markers
from app.generation.llm import LLMError
from app.generation.memory import ChatMemory
from app.generation.prompts import FOLLOWUP_MARKER
from app.retrieval.chunker import chunk_document
from app.retrieval.retriever import is_summary_request
from tests.conftest import FakeLLM

REPORT = ("# Q3 Report\n\nRevenue for Q3 reached $4.2 million, a 24% increase.\n"
          "Operating expenses totaled $2.1 million.\n\n## Risks\n\nSupply chain delays may affect Q4 deliveries.")


def _index(store, make_doc, text=REPORT, doc_id="r1", filename="report.txt"):
    store.add_document({"doc_id": doc_id, "filename": filename, "file_type": "txt", "size_bytes": 1,
                        "page_count": 1, "created_at": "2026-01-01T00:00:00+00:00"},
                       chunk_document(make_doc(doc_id=doc_id, filename=filename, text=text)))


def _run(question, **kwargs):
    events = list(stream_answer(question, **kwargs))
    return events, events[-1][1]


def test_marker_splitter_hides_followups_split_across_tokens():
    splitter = _MarkerSplitter(FOLLOWUP_MARKER)
    text = "Answer text [1].\n" + FOLLOWUP_MARKER + '\n["Q1?", "Q2?"]'
    emitted = "".join(splitter.feed(text[i:i + 3]) for i in range(0, len(text), 3)) + splitter.flush()
    assert emitted == "Answer text [1].\n"
    assert '["Q1?", "Q2?"]' in splitter.tail


def test_citation_normalization():
    assert normalize_citation_markers("A [1, 2]. B [9]. Year [2026].", 2) == "A [1][2]. B. Year [2026]."
    assert is_not_found_answer("I couldn't find this in the uploaded documents. The documents cover X.")


def test_summary_detection():
    assert is_summary_request("Summarize this document")
    assert is_summary_request("what is this report about?")
    assert not is_summary_request("Summarize the risk factors for Q4")


def test_memory_is_bounded_and_strips_citations():
    memory = ChatMemory(max_history_turns=2, max_conversations=2)
    for i in range(5):
        memory.add_turn("c1", f"q{i}", f"answer {i} [1][2].")
    history = memory.get_history("c1")
    assert len(history) == 4 and history[-1]["content"] == "answer 4."
    memory.add_turn("c2", "q", "a")
    memory.add_turn("c3", "q", "a")
    assert memory.get_history("c1") == []  # least recently used conversation evicted


def test_no_documents_message(store):
    _, done = _run("anything?")
    assert "No documents" in done["answer"] and done["citations"] == []


def test_llm_answer_streams_with_citations_and_followups(store, make_doc, use_llm):
    _index(store, make_doc)
    llm = use_llm(FakeLLM(answer="Revenue was $4.2 million [1][7].\n" + FOLLOWUP_MARKER +
                          '\n["What drove growth?", "What are the risks?"]'))
    events, done = _run("What was Q3 revenue?", style=AnswerStyle.CONCISE)

    kinds = [e for e, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and "token" in kinds
    streamed = "".join(d["text"] for e, d in events if e == "token")
    assert FOLLOWUP_MARKER not in streamed
    assert done["answer"] == "Revenue was $4.2 million [1]."  # invalid [7] removed
    assert [c["number"] for c in done["citations"]] == [1]
    assert done["suggested_questions"] == ["What drove growth?", "What are the risks?"]
    assert done["llm_used"] and done["confidence"] in ("High", "Medium", "Low")
    assert "[1] (report.txt" in llm.prompts[-1]


def test_llm_failure_falls_back_to_extractive_answer(store, make_doc, use_llm):
    _index(store, make_doc)
    use_llm(FakeLLM(fail=LLMError("API quota or rate limit exceeded")))
    _, done = _run("What was Q3 revenue?")
    assert not done["llm_used"]
    assert "API quota or rate limit exceeded" in done["answer"]
    assert "$4.2 million" in done["answer"] and done["citations"]


def test_follow_up_question_is_rewritten_with_history(store, make_doc, use_llm):
    _index(store, make_doc)
    llm = use_llm(FakeLLM(answer="Supply chain delays [1].",
                          rewrite='{"standalone": "What risks affect Q4 deliveries?", "queries": []}'))
    _, first = _run("What was Q3 revenue?")
    _, second = _run("and what about Q4?", conversation_id=first["conversation_id"])
    assert second["standalone_question"] == "What risks affect Q4 deliveries?"
    assert "Conversation so far:" in llm.prompts[-1]


def test_unrelated_question_reports_not_found(store, make_doc):
    _index(store, make_doc)
    _, done = _run("Distance from Earth to Mars?")
    assert done["confidence"] == "Insufficient Evidence" and done["citations"] == []


def test_extractive_answer_prefers_matching_sentences(make_doc):
    chunks = chunk_document(make_doc(text=REPORT))
    answer = build_extractive_answer("operating expenses", chunks, AnswerStyle.CONCISE, "no LLM")
    assert "Operating expenses totaled $2.1 million. [1]" in answer
