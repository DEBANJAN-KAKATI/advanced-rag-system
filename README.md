# Advanced RAG System (jstRAG)

A local **Retrieval-Augmented Generation** app: upload documents, ask questions, and get answers that are grounded in your files, with **numbered, clickable citations** that show the exact source passage.

Built with **FastAPI**, **FAISS**, BM25, an optional cross-encoder reranker, and **Gemini** or any **OpenAI-compatible** model (OpenAI, Ollama, LM Studio, vLLM). It works fully offline with a local model, and it still works with no model at all (extractive answers).

---

## Features

- **Formats:** PDF (OCR fallback for scanned pages), DOCX, HTML, TXT/Markdown/RST, CSV, JSON, Python
- **Structure-aware chunking:** splits at paragraph/sentence boundaries, tracks section headings across pages, records page ranges
- **Hybrid retrieval:** vector search + BM25 keyword search, merged with Reciprocal Rank Fusion
- **Reranking:** cross-encoder relevance scores (0–1) filter out unrelated passages
- **Conversation-aware:** follow-ups like *"and what about Q4?"* are rewritten into standalone search queries
- **Whole-document summaries:** "Summarize this document" samples passages across the entire file instead of only the closest matches
- **Streaming answers** with inline `[1]` citations that open the full source passage
- **Answer styles:** Concise / Medium / Detailed (detailed answers also get more context)
- **Scope questions** to selected documents
- **Honest confidence** labels, and an explicit *"I couldn't find this in the uploaded documents"* instead of guessing
- **Graceful degradation:** when the LLM is unavailable (quota, bad key, offline), you get an extractive answer from the most relevant sentences, with the reason shown
- **Safe ingestion:** duplicate detection by content hash, same-name re-uploads replace the old version, upload size limit, no path traversal

---

## Quick start

```bash
pip install -r requirements.txt      # see requirements.txt for a smaller CPU-only torch install
cp .env.example .env                 # then add your GEMINI_API_KEY (optional)
python run.py                        # opens http://localhost:8000
```

Python 3.9+ is required. The data (uploaded files and index) lives in `./data`.

### Choosing a model

| Setup | `.env` | Notes |
|---|---|---|
| **Gemini** (default) | `GEMINI_API_KEY=...` | Free tier available. Uses `gemini-flash-latest` and `gemini-embedding-001`. |
| **Fully local** (Ollama) | `LLM_PROVIDER=openai`<br>`OPENAI_BASE_URL=http://localhost:11434/v1`<br>`LLM_MODEL=llama3.1` | No API key or quota. Embeddings and reranking run locally via sentence-transformers. |
| **OpenAI** | `LLM_PROVIDER=openai`<br>`OPENAI_API_KEY=...`<br>`LLM_MODEL=gpt-4o-mini` | |
| **No LLM** | nothing | Answers are the most relevant extracted sentences, still cited. |

The legacy `api_key_here` file (Gemini key on the first non-comment line) is still read if `GEMINI_API_KEY` is not set.

Every setting is documented in [`.env.example`](.env.example). Changing the embedding model is safe: the index notices and re-embeds your documents on the next start.

---

## How a question is answered

1. **Query rewriting**: the question and recent conversation become a standalone query plus up to two alternative phrasings (one small LLM call).
2. **Retrieval**: all queries are embedded in one call and searched in FAISS; BM25 runs on each query; results are fused with RRF.
3. **Reranking**: a cross-encoder scores each candidate; passages below `MIN_RELEVANCE_SCORE` are dropped.
4. **Generation**: the top passages are numbered `[1]..[n]` and the model must cite them. Passages are treated as untrusted data (prompt-injection guard). The answer streams to the UI, and follow-up suggestions come from the same call.
5. **Post-processing**: invalid citation numbers are removed, only the cited sources are listed, and confidence is derived from reranker relevance.

When the API quota runs out, rate-limited calls are retried briefly (at most `LLM_MAX_RETRY_WAIT_SECONDS`, 20 s by default), and then the extractive fallback answers with a note explaining why.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/upload` | Upload one or more files (multipart field `files`). Per-file status: `indexed`, `duplicate`, `replaced`, `error`. |
| `GET` | `/api/documents` | List indexed documents |
| `GET` | `/api/documents/{doc_id}` | A document's extracted text and chunks |
| `DELETE` | `/api/documents/{doc_id}` | Remove a document |
| `POST` | `/api/chat` | Ask a question, JSON response |
| `POST` | `/api/chat/stream` | Ask a question, Server-Sent Events: `sources`, `token`, `done`, `error` |
| `DELETE` | `/api/conversations/{id}` | Forget a conversation |
| `GET` | `/api/status` | Active LLM / embeddings / reranker and any warnings |

Chat request body:

```json
{ "question": "What was Q3 revenue?", "style": "concise | medium | long",
  "document_ids": ["optional", "filter"], "conversation_id": "optional" }
```

Interactive docs are at `http://localhost:8000/docs`.

---

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

The tests run fully offline (fake embedder and LLM) against a temporary data directory.

---

## Project structure

```text
app/
├── main.py                  # FastAPI app, static UI, background index build
├── api/                     # upload, documents, chat (JSON + SSE), status
├── core/                    # settings (.env), pydantic schemas
├── ingestion/               # pdf, docx, html, txt/csv/json parsers, OCR, cleaning
├── retrieval/
│   ├── chunker.py           # structure-aware chunking
│   ├── embedder.py          # Gemini / local sentence-transformers embeddings
│   ├── keyword.py           # BM25 inverted index
│   ├── vectorstore.py       # FAISS index + document registry, persistence
│   ├── query_rewriter.py    # standalone question + query expansion
│   ├── reranker.py          # RRF + cross-encoder / heuristic reranking
│   └── retriever.py         # hybrid retrieval pipeline
├── generation/
│   ├── llm.py               # Gemini + OpenAI-compatible clients, retries
│   ├── prompts.py           # system prompt, styles
│   ├── answerer.py          # streaming RAG pipeline, extractive fallback
│   ├── citation.py          # citation parsing, confidence
│   └── memory.py            # bounded conversation memory
└── utils/
web/                         # UI (vendored marked + DOMPurify, works offline)
tests/                       # pytest suite
```

### Building a Windows executable

`run.py` supports PyInstaller. Bundle the `web/` folder as data (for example, `--add-data "web;web"`) and place `.env` next to the executable.
