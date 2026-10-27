"""
Tests for the evaluation harness and the retrieval strategies it compares.

They use a deterministic hashing embedder, so no model download or API key is
needed: `python -m pytest tests`.
"""
import hashlib
import re

import numpy as np
import pytest

from app.core import telemetry
from app.core.config import settings
from app.core.pricing import cost_usd
from app.retrieval.chunker import CHUNK_STRATEGIES, chunk_document
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import RetrievalConfig, retrieve_relevant_chunks
from app.retrieval.vectorstore import VectorStore
from evaluation import metrics as M
from evaluation.dataset import Evidence, QAItem, chunk_covers, load_corpus, load_dataset, validate
from evaluation.judges import LexicalJudge, LLMJudge, extract_claims


class HashEmbedder:
    """Bag-of-words hashing embedder: similar texts get similar vectors."""
    name = "hash"

    def embed_texts(self, texts):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms

    def embed_query(self, q):
        return self.embed_texts([q])


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def pep257(corpus):
    return corpus.docs["pep-0257-docstring-conventions.txt"]


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "your_api_key")


def test_golden_dataset_spans_exist_verbatim(corpus):
    items = load_dataset()
    assert validate(items, corpus) == []
    assert sum(i.answerable for i in items) >= 70
    assert any(not i.answerable for i in items)


@pytest.mark.parametrize("strategy", CHUNK_STRATEGIES)
def test_chunkers_keep_all_content(strategy, pep257):
    chunks = chunk_document(pep257, strategy=strategy, chunk_size=150, chunk_overlap=30,
                            embed_fn=HashEmbedder().embed_texts)
    assert len(chunks) > 3
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    doc_words = set(re.findall(r"\w+", pep257["raw_text"]))
    chunk_words = set().union(*(re.findall(r"\w+", c["text"]) for c in chunks))
    assert doc_words <= chunk_words


def test_fixed_and_sentence_chunks_respect_size(pep257):
    for strategy in ("fixed", "sentence"):
        chunks = chunk_document(pep257, strategy=strategy, chunk_size=150, chunk_overlap=30)
        longest_sentence = 600  # a single sentence may exceed the target on its own
        assert max(len(c["text"]) for c in chunks) <= 150 * 4 + longest_sentence


def test_structure_overlap_zero_does_not_duplicate(pep257):
    chunks = chunk_document(pep257, strategy="structure", chunk_size=150, chunk_overlap=0)
    total = sum(len(c["text"]) for c in chunks)
    assert total < len(pep257["raw_text"]) * 1.05


def test_chunk_sections_follow_headings(pep257):
    chunks = chunk_document(pep257, strategy="sentence", chunk_size=150, chunk_overlap=0)
    trim = [c for c in chunks if "def trim(docstring)" in c["text"]]
    assert trim and trim[0]["section"] == "Handling Docstring Indentation"


def _item(*spans, mode="all"):
    return QAItem(id="t", question="q", type="lexical",
                  evidence=[Evidence(doc="d.txt", text=s) for s in spans], evidence_mode=mode)


def _chunk(cid, text):
    return {"chunk_id": cid, "filename": "d.txt", "text": text}


def test_chunk_covers_handles_split_spans_and_other_docs():
    ev = Evidence(doc="d.txt", text="alpha beta gamma delta epsilon zeta eta theta")
    assert chunk_covers(_chunk("a", "xx alpha beta gamma delta epsilon zeta yy"), ev)  # 4/6 trigrams
    assert not chunk_covers(_chunk("b", "theta eta zeta"), ev)
    assert not chunk_covers({"chunk_id": "c", "filename": "other.txt", "text": ev.text}, ev)


def test_retrieval_metrics_values():
    item = _item("the first fact is here", "a second fact lives there")
    ranked = [_chunk("1", "noise"), _chunk("2", "the first fact is here"), _chunk("3", "noise again"),
              _chunk("4", "a second fact lives there")]
    m = M.retrieval_metrics(item, ranked, n_relevant_in_index=2)
    assert m["hit@1"] == 0 and m["hit@3"] == 1
    assert m["evidence_recall@3"] == 0.5 and m["evidence_recall@5"] == 1.0
    assert m["mrr"] == 0.5
    assert m["precision@5"] == pytest.approx(2 / 5)
    ideal = 1 + 1 / np.log2(3)
    assert m["ndcg@10"] == pytest.approx((1 / np.log2(3) + 1 / np.log2(5)) / ideal)

    any_item = _item("the first fact is here", "a second fact lives there", mode="any")
    assert M.retrieval_metrics(any_item, ranked, 2)["evidence_recall@3"] == 1.0


def test_token_budget_recall_penalises_big_chunks():
    item = _item("needle sentence that answers")
    big = [_chunk("1", "filler " * 900 + "needle sentence that answers")]
    small = [_chunk("1", "needle sentence that answers")]
    assert M.retrieval_metrics(item, big, 1)["recall@1000tok"] == 0.0
    assert M.retrieval_metrics(item, small, 1)["recall@1000tok"] == 1.0


def test_lexical_judge_flags_unsupported_and_changed_numbers():
    context = ("Limit all lines to a maximum of 79 characters. Docstrings and comments "
               "should be limited to 72 characters. Use 4 spaces per indentation level.")
    judge = LexicalJudge()
    good = judge.faithfulness("Limit all lines to a maximum of 79 characters [Doc: pep8.txt].", context)
    assert good["faithfulness"] == 1.0
    wrong_number = judge.faithfulness("Limit all lines to a maximum of 99 characters.", context)
    assert wrong_number["faithfulness"] == 0.0
    injected = judge.faithfulness("Use 4 spaces per indentation level. The walrus operator was "
                                  "introduced in Python 3.8 by assignment expressions.", context)
    assert injected["faithfulness"] == 0.5 and len(injected["unsupported"]) == 1


def test_claim_extraction_skips_markup_and_offline_notes():
    answer = ("### Heading\n**pep-0008-style-guide.txt** (Page 1, Maximum Line Length):\n"
              "Limit all lines to a maximum of 79 characters [Doc: a.txt, Page: 1].\n"
              "*⚠️ Offline mode — Gemini API unavailable.*")
    assert extract_claims(answer) == ["Limit all lines to a maximum of 79 characters ."]


class _FakeResponse:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = type("U", (), {"prompt_token_count": 1000, "candidates_token_count": 50,
                                             "thoughts_token_count": None})()


class _FakeClient:
    def __init__(self, replies):
        replies = list(replies)
        self.models = type("M", (), {"generate_content": lambda _self, model, contents: _FakeResponse(replies.pop(0))})()


def test_llm_judge_parses_claims_and_tracks_its_own_cost():
    client = _FakeClient(['{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}',
                          '```json\n{"verdict": "partial"}\n```'])
    judge = LLMJudge(model="gemini-2.0-flash", client=client)
    out = judge.judge("q", "answer", [{"text": "ctx"}], reference="ref")
    assert out["faithfulness"] == 0.5 and out["unsupported"] == ["b"]
    assert out["correctness"] == 0.5
    assert len(judge.usage) == 2 and judge.usage[0]["llm_input_tokens"] == 1000


def test_telemetry_costs_and_nesting():
    assert cost_usd("gemini-2.0-flash", 1_000_000, 1_000_000) == pytest.approx(0.50)
    assert cost_usd("unknown-model", 10) is None
    with telemetry.trace() as outer:
        telemetry.record_llm_call("answer", "gemini-2.0-flash", 2000, 200)
        telemetry.record_embedding("local-onnx", "all-MiniLM-L6-v2", 3, 30)
        with telemetry.trace() as inner:
            assert inner is outer
            with telemetry.stage("rerank"):
                pass
    s = outer.summary()
    assert s["llm_calls"] == 1 and s["embedding_tokens"] == 30
    assert s["estimated_cost_usd"] == pytest.approx((2000 * 0.10 + 200 * 0.40) / 1e6)
    assert "rerank" in s["stages_ms"]
    telemetry.record_llm_call("answer", "x", 1, 1)  # no active trace: silently ignored


@pytest.fixture(scope="module")
def small_store(corpus):
    store = VectorStore(embedder=HashEmbedder(), persist=False)
    for name in ("pep-0257-docstring-conventions.txt", "pep-0405-virtual-environments.txt"):
        store.add_chunks(chunk_document(corpus.docs[name], strategy="sentence", chunk_size=150, chunk_overlap=30))
    return store


@pytest.mark.parametrize("strategy", ["none", "keyword", "mmr", "sentence-maxsim"])
def test_rerank_strategies_return_topk_and_record_strategy(strategy, small_store):
    cfg = RetrievalConfig(use_query_rewrite=False, rerank_strategy=strategy)
    with telemetry.trace() as t:
        results, queries = retrieve_relevant_chunks("What does the --clear option do when creating a venv?",
                                                    top_k=5, store=small_store, config=cfg)
    assert queries == ["What does the --clear option do when creating a venv?"]
    assert len(results) == 5 and len({c["chunk_id"] for c, _ in results}) == 5
    assert t.annotations["rerank_strategy_used"] == strategy
    assert {"embed_query", "bm25", "fusion", "rerank"} <= set(t.stages)


def test_cross_encoder_fallback_is_visible(small_store, monkeypatch):
    monkeypatch.setattr("app.retrieval.reranker.reranker._get_cross_encoder", lambda: None)
    with telemetry.trace() as t:
        retrieve_relevant_chunks("pyvenv.cfg home key", top_k=3, store=small_store,
                                 config=RetrievalConfig(use_query_rewrite=False, rerank_strategy="cross-encoder"))
    assert t.annotations["rerank_strategy_used"] == "keyword"


def test_dense_only_and_bm25_only(small_store):
    for cfg in (RetrievalConfig(use_bm25=False, use_query_rewrite=False, rerank_strategy="none"),
                RetrievalConfig(use_dense=False, use_query_rewrite=False, rerank_strategy="none")):
        results, _ = retrieve_relevant_chunks("pyvenv.cfg home key", top_k=3, store=small_store, config=cfg)
        assert results
    with pytest.raises(ValueError):
        retrieve_relevant_chunks("q", store=small_store, config=RetrievalConfig(use_dense=False, use_bm25=False))


def test_bm25_cache_gives_identical_results(small_store):
    chunks = small_store.get_all_chunks()
    cached, uncached = Reranker(), Reranker()
    uncached.cache_bm25 = False
    for q in ("virtual environment isolation", "docstring indentation"):
        assert cached.bm25_search(q, chunks) == uncached.bm25_search(q, chunks)


def test_in_memory_store_does_not_touch_disk(tmp_path, corpus):
    store = VectorStore(embedder=HashEmbedder(), persist=False, save_dir=tmp_path)
    store.add_chunks(chunk_document(corpus.docs["pep-0257-docstring-conventions.txt"]))
    assert list(tmp_path.iterdir()) == []
    ids = [c["chunk_id"] for c in store.get_all_chunks()[:2]]
    assert store.get_vectors(ids).shape == (2, 256)


def test_generate_answer_reports_usage_offline(small_store, no_llm):
    from app.generation.answerer import generate_answer
    resp = generate_answer("What is a docstring?", store=small_store,
                           retrieval_config=RetrievalConfig(use_query_rewrite=False, rerank_strategy="none"))
    assert resp.usage["llm_calls"] == 0
    assert resp.usage["estimated_cost_usd"] == 0.0
    assert "retrieval" in resp.usage["stages_ms"]
    assert all(s["chunk_id"] for s in resp.sources_used)
