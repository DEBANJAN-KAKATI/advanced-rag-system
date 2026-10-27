import pytest
from app.retrieval.chunker import chunk_document
from app.retrieval.embedder import EmbeddingError
from app.retrieval.keyword import BM25Index, tokenize
from app.retrieval.vectorstore import VectorStore
from tests.conftest import FakeEmbedder


def _add(store, make_doc, doc_id, filename, text):
    chunks = chunk_document(make_doc(doc_id=doc_id, filename=filename, text=text))
    store.add_document({"doc_id": doc_id, "filename": filename, "file_type": "txt", "size_bytes": len(text),
                        "page_count": 1, "created_at": "2026-01-01T00:00:00+00:00"}, chunks)
    return chunks


def test_bm25_scores_single_document_corpus():
    # rank_bm25's Okapi IDF went negative here, so a one-chunk corpus never matched.
    index = BM25Index(["Revenue for Q3 reached $4.2 million."])
    assert index.scores("What was the revenue?")[0] > 0


def test_tokenizer_strips_punctuation_and_stopwords():
    assert tokenize("What was the Revenue, in Q3?") == ["revenue", "q3"]
    assert "4.2" in tokenize("$4.2 million")


def test_add_search_delete_without_reembedding(tmp_path, make_doc):
    emb = FakeEmbedder()
    store = VectorStore(tmp_path, embedder=emb)
    _add(store, make_doc, "a", "a.txt", "Apples are red fruit grown in orchards.")
    _add(store, make_doc, "b", "b.txt", "Bananas are yellow tropical fruit.")
    assert store.vectors_ready
    embedded_before = emb.embedded_texts

    results = store.vector_search(emb.embed_queries(["yellow bananas"]), top_k=2)[0]
    assert results[0][0]["doc_id"] == "b"

    assert store.delete_document("a")
    assert emb.embedded_texts == embedded_before + 1  # only the query above; no re-embedding
    assert store.vectors_ready and len(store.chunks) == 1
    assert store.vector_search(emb.embed_queries(["apples"]), top_k=5)[0][0][0]["doc_id"] == "b"


def test_document_filter_applies_to_both_searches(tmp_path, make_doc):
    emb = FakeEmbedder()
    store = VectorStore(tmp_path, embedder=emb)
    _add(store, make_doc, "a", "a.txt", "Shared topic: pricing for plan A.")
    _add(store, make_doc, "b", "b.txt", "Shared topic: pricing for plan B.")
    vec = store.vector_search(emb.embed_queries(["pricing"]), top_k=5, doc_filter=["b"])[0]
    kw = store.keyword_search("pricing", top_k=5, doc_filter=["b"])
    assert {c["doc_id"] for c, _ in vec} == {"b"} == {c["doc_id"] for c, _ in kw}


def test_persistence_and_reindex_on_model_change(tmp_path, make_doc):
    store = VectorStore(tmp_path, embedder=FakeEmbedder())
    _add(store, make_doc, "a", "a.txt", "Persistent knowledge about turbines.")

    reloaded = VectorStore(tmp_path, embedder=FakeEmbedder())
    assert reloaded.vectors_ready and reloaded.get_document("a")["filename"] == "a.txt"

    # A different embedding model must never be mixed into the old index.
    new_model = FakeEmbedder(dim=32, name="other")
    switched = VectorStore(tmp_path, embedder=new_model)
    assert not switched.vectors_ready
    assert switched.ensure_index() and switched.index.d == 32
    assert new_model.embedded_texts == len(switched.chunks)


def test_failed_embedding_leaves_store_unchanged(tmp_path, make_doc):
    emb = FakeEmbedder()
    store = VectorStore(tmp_path, embedder=emb)
    _add(store, make_doc, "a", "a.txt", "Existing document text.")
    emb.fail = True
    with pytest.raises(EmbeddingError):
        _add(store, make_doc, "b", "b.txt", "New document text.")
    assert list(store.documents) == ["a"] and store.vectors_ready


def test_legacy_metadata_is_migrated(tmp_path):
    import json
    (tmp_path / "metadata.json").write_text(json.dumps([
        {"chunk_id": "x_c0", "doc_id": "x", "filename": "old.pdf", "page": 2, "section": "General",
         "text": "Legacy chunk text.", "char_count": 18},
    ]))
    emb = FakeEmbedder()
    store = VectorStore(tmp_path, embedder=emb)
    assert store.get_document("x")["page_count"] == 2
    assert store.ensure_index() and store.vectors_ready


def test_failed_reindex_is_not_retried_on_every_call(tmp_path, make_doc):
    store = VectorStore(tmp_path, embedder=FakeEmbedder())
    _add(store, make_doc, "a", "a.txt", "Some text to embed.")
    broken = FakeEmbedder(name="broken")
    broken.fail = True
    store2 = VectorStore(tmp_path, embedder=broken)
    assert not store2.ensure_index() and "outage" in store2.last_index_error
    broken.fail = False
    assert not store2.ensure_index()          # within the cooldown: no new attempt
    assert store2.ensure_index(force=True)    # explicit retry works
