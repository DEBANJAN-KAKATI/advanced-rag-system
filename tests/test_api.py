import json
import pytest
from fastapi.testclient import TestClient
from app.core.config import settings
from app.main import app


@pytest.fixture
def client(store):
    return TestClient(app)


def _upload(client, *files):
    return client.post("/api/upload", files=[("files", f) for f in files]).json()


def test_upload_list_duplicate_and_delete(client):
    content = b"# Handbook\n\nVacation policy: employees get 25 days of paid leave per year."
    first = _upload(client, ("handbook.md", content, "text/markdown"))[0]
    assert first["success"] and first["status"] == "indexed"

    dup = _upload(client, ("copy.md", content, "text/markdown"))[0]
    assert dup["status"] == "duplicate" and dup["doc_id"] == first["doc_id"]

    docs = client.get("/api/documents").json()
    assert docs["total_count"] == 1 and docs["documents"][0]["filename"] == "handbook.md"
    assert client.get(f"/api/documents/{first['doc_id']}").json()["chunk_count"] >= 1

    stored = settings.UPLOAD_DIR / f"{first['doc_id']}.md"
    assert stored.exists()
    assert client.delete(f"/api/documents/{first['doc_id']}").json()["success"]
    assert not stored.exists()
    assert client.get("/api/documents").json()["total_count"] == 0
    assert client.delete(f"/api/documents/{first['doc_id']}").status_code == 404


def test_reupload_with_same_name_replaces_previous_version(client):
    old = _upload(client, ("notes.txt", b"Version one of the notes.", "text/plain"))[0]
    new = _upload(client, ("notes.txt", b"Version two of the notes.", "text/plain"))[0]
    assert new["status"] == "replaced"
    ids = [d["doc_id"] for d in client.get("/api/documents").json()["documents"]]
    assert ids == [new["doc_id"]] and old["doc_id"] not in ids


def test_upload_rejections(client, monkeypatch):
    results = _upload(
        client,
        ("../../etc/evil.exe", b"x", "application/octet-stream"),
        ("blank.txt", b"   ", "text/plain"),
    )
    assert [r["success"] for r in results] == [False, False]
    assert results[0]["filename"] == "evil.exe"  # path components stripped
    assert "No readable text" in results[1]["message"]

    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0)
    big = _upload(client, ("big.txt", b"x" * 10, "text/plain"))[0]
    assert not big["success"] and "limit" in big["message"]
    assert not any(p.name.startswith(".incoming") for p in settings.UPLOAD_DIR.iterdir())


def test_chat_json_and_stream(client):
    _upload(client, ("faq.txt", b"The office opens at 9am and closes at 6pm on weekdays.", "text/plain"))

    data = client.post("/api/chat", json={"question": "When does the office open?", "style": "concise"}).json()
    assert "9am" in data["answer"] and data["citations"][0]["filename"] == "faq.txt"
    assert data["conversation_id"]

    with client.stream("POST", "/api/chat/stream", json={"question": "When does the office close?"}) as r:
        body = "".join(r.iter_text())
    events = [block.split("\n") for block in body.strip().split("\n\n")]
    names = [lines[0].removeprefix("event: ") for lines in events]
    assert names[0] == "sources" and names[-1] == "done"
    done = json.loads(events[-1][1].removeprefix("data: "))
    assert "6pm" in done["answer"]


def test_streamed_answer_reports_usage(client, use_llm):
    """Streaming advances the pipeline in worker threads; telemetry must survive that."""
    from tests.conftest import FakeLLM
    _upload(client, ("faq.txt", b"The office opens at 9am and closes at 6pm on weekdays.", "text/plain"))
    use_llm(FakeLLM(answer="It opens at 9am [1].",
                    rewrite='{"standalone": "When does the office open?", "queries": ["office hours"]}'))
    with client.stream("POST", "/api/chat/stream", json={"question": "When does the office open?"}) as r:
        body = "".join(r.iter_text())
    done = json.loads(body.strip().split("\n\n")[-1].split("\n")[1].removeprefix("data: "))
    usage = done["usage"]
    assert usage["llm_calls"] == 2  # query rewrite + answer
    assert {"query_rewrite", "retrieval", "generation", "rerank"} <= set(usage["stages_ms"])
    assert usage["annotations"]["rerank_strategy_used"] == "keyword"


def test_chat_validation(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={"question": "   "}).status_code == 400


def test_status_reports_degraded_modes(client):
    status = client.get("/api/status").json()
    assert status["llm_available"] is False
    assert any("No LLM configured" in w for w in status["warnings"])


def test_frontend_is_served(client):
    html = client.get("/").text
    assert "vendor/purify.min.js" in html
    assert client.get("/vendor/marked.min.js").status_code == 200
