# Evaluation harness

Measures three things for any combination of chunking and retrieval strategy:

| What | How |
|---|---|
| **Retrieval accuracy** | hit@k, evidence recall@k, MRR, nDCG@10, precision@5, and recall within a fixed LLM token budget, against hand-labelled evidence spans |
| **Faithfulness**: do answers stick to the source? | Share of answer claims supported by the retrieved context. Two judges: an offline lexical judge, and an LLM claim-verification judge (Gemini). Also scores correctness against a reference answer and abstention on unanswerable questions |
| **Cost and speed per query** | Per-stage latency (embed, vector search, BM25, fusion, rerank, generation) and token usage from the app's own telemetry, priced with `app/core/pricing.py`. Offline runs project what the same query would cost on Gemini |

The final report is in [`docs/EVALUATION_REPORT.md`](../docs/EVALUATION_REPORT.md).

## Quick start

```bash
pip install -r requirements.txt -r evaluation/requirements.txt

python -m evaluation.validate_dataset          # every evidence span exists verbatim in the corpus
python -m pytest tests                          # unit tests (no model download, no API key)

# Full offline grid: 9 chunking x 6 retrieval configs (~15 min on 4 CPUs)
python -m evaluation.run --suite full --generate --calibrate-judge lexical --name my-run
cat evaluation/results/my-run/summary.md
```

The first run downloads the ONNX build of `all-MiniLM-L6-v2` (~80 MB) into `data/models/`. This is the same model the app uses as its local embedding fallback, and the download is checked against a pinned SHA-256. Nothing the harness does touches the app's persisted index in `data/vectorstore/`.

### With Gemini (measured cost, LLM-judged faithfulness)

Put a key in `api_key_here` (or set `GEMINI_API_KEY`), then:

```bash
# Measured end-to-end tokens, dollars and latency for the app's real generate_answer()
python -m evaluation.run --chunking structure-500 sentence-250 --retrieval hybrid-rrf hybrid+keyword \
    --generate --judge llm --name gemini-e2e

# The app's full default pipeline (query rewriting + cross-encoder). Needs sentence-transformers
# and access to huggingface.co for the cross-encoder model.
python -m evaluation.run --retrieval hybrid+rewrite+cross-encoder --chunking structure-500 --generate --judge llm

# How far to trust each judge: score answers whose faithfulness is known
python -m evaluation.run --suite none --calibrate-judge lexical llm --name judge-cal
```

`--embedder app` uses the app's configured embedder (Gemini when a key is set) instead of local MiniLM.

## Files

| Path | Purpose |
|---|---|
| `corpus/` | 8 Python Enhancement Proposals (public domain), about 52k words. Rebuild with `python -m evaluation.build_corpus` |
| `datasets/pep_qa.jsonl` | 81 questions: 75 answerable (lexical, paraphrase, multi-span, cross-document) and 6 unanswerable |
| `dataset.py` | Loads the dataset and corpus. Defines when a chunk "contains" an evidence span, independent of how the text was chunked |
| `metrics.py` | Retrieval metrics, bootstrap confidence intervals, paired bootstrap significance test |
| `judges.py` | Lexical and LLM faithfulness/correctness judges, plus judge calibration |
| `backends.py` | ONNX MiniLM embedder |
| `run.py` | Strategy configs, the experiment loop, and output (`summary.md`, `summary.json`, `per_query.jsonl`) |
| `results/` | Committed results that the report cites |

## Dataset format

```json
{"id": "q026", "type": "multi_span",
 "question": "Can globally installed packages be imported from a virtual environment, and if the same package exists in both places which one wins?",
 "evidence": [{"doc": "pep-0405-virtual-environments.txt", "text": "By default, a virtual environment is entirely isolated from ..."},
              {"doc": "pep-0405-virtual-environments.txt", "text": "Thus system-installed packages will still be importable, but ..."}],
 "evidence_mode": "all",
 "reference_answer": "Not by default ..."}
```

Relevance is labelled with **evidence spans, not chunk IDs**, so every chunking strategy is scored against the same ground truth. A retrieved chunk counts as relevant to a span if it contains the span, or at least 50% of the span's word trigrams when a chunk boundary cuts through it. `evidence_mode: "any"` marks spans that are interchangeable (the same fact stated in two documents). To add questions for your own documents, put the files in a corpus directory and copy verbatim evidence; `validate_dataset` rejects anything that isn't verbatim.
