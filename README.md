#  Advanced RAG System

A production-grade, multi-format **Retrieval-Augmented Generation (RAG)** application built in **Python** with **FastAPI** and a modern web interface.  
This system is designed to help users deeply understand documents by asking questions over uploaded files and receiving grounded, well-structured answers.

It supports:
- **PDF**
- **DOC / DOCX**
- **HTML**
- **TXT**
- other readable text-based files

It also provides **answer length control**:
- **Concise**
- **Medium**
- **Detailed**

With advanced retrieval, strong grounding, fallback handling, and a polished UI, this project goes far beyond a basic document chatbot.

---

##  Key Highlights

- Multi-format document ingestion
- Intelligent PDF extraction with OCR fallback
- Hybrid retrieval using semantic + keyword search
- Query rewriting for better search quality
- Reranking for more accurate context selection
- Gemini-powered answer generation
- Citation-aware responses
- Concise / Medium / Detailed answer modes
- Conversation memory for follow-up questions
- Clickable AI-generated follow-up suggestions
- Offline fallback when API limits are reached
- Evaluation harness: retrieval accuracy, faithfulness, and cost/latency per query ([report](docs/EVALUATION_REPORT.md))
- Beautiful interactive web dashboard

---

##  What This System Does

This application allows you to upload documents and ask natural language questions about them.

Example questions:
- “Summarize this report.”
- “What does the document say about pricing?”
- “Explain the methodology in detail.”
- “Give me a concise answer.”
- “What are the main conclusions?”

The system reads the content, retrieves the most relevant sections, and generates an answer grounded in the uploaded files.

---

##  How to Run It

You have two ways to run the application:

### Option 1: Running the Executable (Windows)
If you just want to run the app without installing Python or any dependencies:
1. Double-click the `jstRAG.exe` file in the root folder.
2. A command prompt window will open showing the server starting.
3. Open your web browser and go to `http://localhost:8000`.

### Option 2: Running from Source
If you want to run it via Python:
1. Ensure you have Python 3.9+ installed.
2. Open a terminal in the project directory.
3. Install dependencies (optional if already installed): `pip install -r requirements.txt`
4. Run the server: `python run.py`
5. Open your web browser and go to `http://localhost:8000`.

---

##  Where to Place the API Key

This system uses Google's Gemini AI to generate answers. You need to provide an API key.

1. Open the file named `api_key_here` located in the root of the project directory.
2. Paste your Gemini API key directly into this file, just below the comment line.
3. Save the file.
*(Note: Do not use quotes or add extra spaces. The system will automatically read it and ignore comments or blank lines).*

---

## What Happens When Your Quota is Finished?

Google's free-tier Gemini API has strict daily and per-minute rate limits. Because this advanced RAG system makes multiple calls per question (for query rewriting, embeddings, and answer generation), it is easy to exhaust the free quota.

**Here is how the system intelligently handles rate limits:**

1. **Automatic Retries:** If you hit a per-minute limit, the system detects the "Resource Exhausted" error, reads exactly how long it needs to wait, pauses gracefully, and retries your request up to 3 times in the background.
2. **Offline Fallback Mode:** If your daily quota is fully exhausted, the system **will not break or throw an ugly error**. Instead, it seamlessly switches to an "Offline Fallback" mode.

### How It Answers in Offline Fallback Mode
When the API is unreachable, the system relies strictly on its local search capabilities to give you the raw context directly from your documents. 

The fallback answer respects your selected style:
- **Concise:** Shows the single most relevant passage (the first 2-3 sentences only) and its page number.
- **Medium:** Shows the top 2-3 relevant passages (around 5 sentences each) with their respective filenames and page numbers.
- **Detailed:** Dumps **all** relevant text chunks found across your documents, fully untruncated, and neatly grouped by Document Name and Page Number.

You will also see a warning note at the bottom of the answer: *"⚠️ Offline mode — Gemini API unavailable. Showing direct document extract."* Once your quota resets (usually at midnight), the system automatically resumes generating AI-synthesized answers.

---

##  Smart Follow-up Questions

After every answer, the system generates **4 clickable suggestion chips** at the bottom of the chat bubble. 
- **Online Mode:** Gemini AI analyzes the answer and predicts what you might want to ask next (e.g., asking for an example, a comparison, or advanced concepts).
- **Offline Mode:** The system uses the internal metadata of the retrieved documents (like section headings) to suggest relevant follow-up questions.

Clicking any of these chips will instantly send it as your next question!

---

##  Evaluation

The `evaluation/` package measures the pipeline on a labelled question set built from public-domain Python PEPs:

- **Retrieval accuracy**: hit@k, evidence recall@k, MRR, nDCG, and recall within a fixed token budget
- **Faithfulness**: share of answer claims supported by the retrieved sources (offline lexical judge, or Gemini as judge), plus abstention on unanswerable questions
- **Cost and speed per query**: per-stage latency and token usage, priced in dollars

```bash
python -m evaluation.run --suite full --generate --calibrate-judge lexical
```

Every `/api/chat` response now includes a `usage` object with per-stage latency (ms), LLM and embedding token counts, and estimated cost. Chunking and reranking strategies can be switched with `CHUNK_STRATEGY` (`structure`, `fixed`, `sentence`, `semantic`) and `RERANK_STRATEGY` (`cross-encoder`, `keyword`, `none`, `mmr`, `sentence-maxsim`); `ENABLE_QUERY_REWRITE=false` turns off the extra LLM call.

Results and recommendations: [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md). Harness usage: [evaluation/README.md](evaluation/README.md).

---

##  Project Structure

```text
RAG/
├── api_key_here
├── jstRAG.exe
├── jstRAG.spec
├── requirements.txt
├── run.py
├── sample_report.txt
├── test_rag.py
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── chat.py
│   │   ├── documents.py
│   │   └── upload.py
│   ├── core/
│   │   ├── config.py
│   │   └── schemas.py
│   ├── generation/
│   │   ├── answerer.py
│   │   ├── citation.py
│   │   ├── memory.py
│   │   └── prompts.py
│   ├── ingestion/
│   │   ├── cleaner.py
│   │   ├── docx.py
│   │   ├── html.py
│   │   ├── loader.py
│   │   ├── ocr.py
│   │   ├── pdf.py
│   │   └── txt.py
│   ├── retrieval/
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── query_rewriter.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   └── vectorstore.py
│   └── utils/
│       ├── filetypes.py
│       └── logging.py
└── web/
    ├── app.js
    ├── index.html
    └── style.css
```