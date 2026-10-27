import numpy as np
import pytest
from app.core.schemas import AnswerStyle
from app.retrieval import retriever as retriever_module
from app.retrieval.chunker import chunk_document
from app.retrieval.reranker import Reranker, reciprocal_rank_fusion
from app.retrieval.vectorstore import VectorStore
from tests.conftest import FakeEmbedder, install_store


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, show_progress_bar=False):
        return np.array([self.scores(q, t) for q, t in pairs])


def _chunks(*texts):
    return [{"chunk_id": f"c{i}", "doc_id": "d", "filename": "f.txt", "text": t} for i, t in enumerate(texts)]


@pytest.mark.parametrize("raw_high, raw_low", [(6.0, -8.0), (0.98, 0.01)])
def test_cross_encoder_scores_are_probabilities(monkeypatch, raw_high, raw_low):
    r = Reranker()
    monkeypatch.setattr(r, "_get_cross_encoder",
                        lambda: FakeCrossEncoder(lambda q, t: raw_high if "match" in t else raw_low))
    ranked, scorer = r.rerank("query", _chunks("nothing", "a match"), strategy="cross-encoder")
    assert scorer == "cross-encoder"
    assert ranked[0][0]["chunk_id"] == "c1"
    assert ranked[0][1] > 0.9 and ranked[1][1] < 0.05  # never double-sigmoided


def test_keyword_scores_stay_in_range():
    ranked, scorer = Reranker().rerank("revenue growth", _chunks("revenue growth revenue", "other"),
                                       strategy="keyword")
    assert scorer == "keyword"
    assert ranked[0][0]["chunk_id"] == "c0"
    assert all(0.0 <= s <= 1.0 for _, s in ranked)


def test_unavailable_cross_encoder_falls_back_visibly(monkeypatch):
    from app.core import telemetry
    r = Reranker()
    monkeypatch.setattr(r, "_get_cross_encoder", lambda: None)
    with telemetry.trace() as t:
        _, scorer = r.rerank("revenue", _chunks("revenue", "other"), strategy="cross-encoder")
    assert scorer == "keyword" and t.annotations["rerank_strategy_used"] == "keyword"


def test_rrf_rewards_agreement():
    a, b, c = _chunks("a", "b", "c")
    fused = reciprocal_rank_fusion([[(a, 1), (b, 1)], [(b, 1), (c, 1)]])
    assert fused[0][0]["chunk_id"] == "c1"


def _setup_store(tmp_path, monkeypatch):
    store = install_store(VectorStore(tmp_path / "vs", embedder=FakeEmbedder()), tmp_path, monkeypatch)
    text = ("## Pricing\n\nThe Pro plan costs $20 per month and includes priority support.\n\n"
            "## Security\n\nAll data is encrypted at rest with AES-256 and in transit with TLS 1.3.")
    doc = {"doc_id": "d1", "filename": "product.md", "pages": [{"page": 1, "text": text}]}
    store.add_document({"doc_id": "d1", "filename": "product.md", "file_type": "md", "size_bytes": 1,
                        "page_count": 1, "created_at": "2026-01-01"}, chunk_document(doc, chunk_size=120, chunk_overlap=0))
    return store


def test_hybrid_retrieval_finds_relevant_section(tmp_path, monkeypatch):
    _setup_store(tmp_path, monkeypatch)
    result = retriever_module.retrieve_relevant_chunks("How much does the Pro plan cost?", style=AnswerStyle.CONCISE)
    assert result.mode == "hybrid"
    assert "$20" in result.chunks[0][0]["text"]


def test_cross_encoder_threshold_flags_unrelated_questions(tmp_path, monkeypatch):
    _setup_store(tmp_path, monkeypatch)
    monkeypatch.setattr(retriever_module.reranker, "_get_cross_encoder",
                        lambda: FakeCrossEncoder(lambda q, t: -9.0))  # nothing is relevant
    result = retriever_module.retrieve_relevant_chunks(
        "Is the data encrypted?", config=retriever_module.RetrievalConfig(rerank_strategy="cross-encoder"))
    assert result.scorer == "cross-encoder" and result.low_relevance
    assert len(result.chunks) <= 3


def test_summary_request_spreads_across_document(tmp_path, monkeypatch):
    _setup_store(tmp_path, monkeypatch)
    result = retriever_module.retrieve_relevant_chunks("Summarize this document")
    assert result.mode == "summary"
    assert {c["section"] for c, _ in result.chunks} == {"Pricing", "Security"}
