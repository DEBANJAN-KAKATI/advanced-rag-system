# Chunking and reranking: evaluation report

**TL;DR**

- **Chunk smaller.** 250-token chunks retrieve as well as the current 500-token default at top-5 (0.820 vs 0.833 evidence recall, not a significant difference). With the context budget held fixed they are much better: 0.807 vs 0.560 recall within 1,000 tokens. That budget difference is significant (p < 0.001). They also cut the answer prompt by 43% (1,626 vs 2,838 tokens) and the projected cost by 31%.
- **Keep hybrid retrieval.** Dense-only retrieval recall@5 is 0.687 and BM25-only is 0.760; fusing them with RRF reaches 0.833 (p = 0.001 vs dense-only). Each retriever covers the other's failure mode.
- **Rerankers are not interchangeable.**
  - A cheap keyword-overlap reranker adds about 5 points of recall@5 for 3 ms.
  - A sentence-level embedding reranker gives the best ranking (MRR 0.754 vs 0.608, p = 0.005), but costs 1.6 s per query on CPU.
  - MMR (maximal marginal relevance) makes results significantly worse (0.673, p = 0.002).
- **Two problems the harness found and fixed:**
  - BM25 was rebuilding its index on every query; caching it cut that stage from 22 ms to 1 ms.
  - When the cross-encoder model fails to load, the app silently falls back to keyword reranking. Which reranker actually ran is now recorded.

Raw results: [`evaluation/results/full-offline/summary.md`](../evaluation/results/full-offline/summary.md). Reproduce them with `python -m evaluation.run --suite full --generate --calibrate-judge lexical`.

## Setup

| | |
|---|---|
| Corpus | 8 Python PEPs (public domain), about 52k words of long, sectioned technical prose with code. Some pairs of documents overlap in vocabulary on purpose (PEP 8 / PEP 257, PEP 484 / PEP 526) |
| Questions | 81 total. 75 answerable: 45 paraphrase, 18 lexical, 8 multi-span, 4 cross-document. 6 unanswerable but plausible. Labels are verbatim evidence spans, machine-checked against the corpus, so every chunking strategy is scored against the same ground truth |
| Retrieval metrics | hit@k, evidence recall@k, MRR, nDCG@10, precision@5, and **recall within a token budget**. Ranks are scored to top-10; the app sends the top 5 to the LLM |
| Statistics | 95% bootstrap CIs; paired bootstrap p-values against the default (`structure-500` + hybrid RRF) |
| Embeddings | `all-MiniLM-L6-v2`, the app's local fallback model, run via ONNX. Its context window is 256 wordpieces |
| Cost | Projected from the exact prompt the app would build, priced at `gemini-2.0-flash` ($0.10 / $0.40 per M input/output tokens) and `gemini-embedding-001` ($0.15/M). Output length is assumed to be 250 tokens (medium style) |
| Hardware | 4-vCPU cloud container; latencies are p50 over 81 queries |

**Not measured in this run.** This environment had no Gemini key, and its network policy blocked huggingface.co, so these results come from a limited setup:

- The cross-encoder reranker and query rewriting were not run.
- Answers came from the app's offline extractive fallback, not Gemini.

The harness supports all of these (`--retrieval hybrid+rewrite+cross-encoder --generate --judge llm`), and running them is the obvious next step.

## Chunking

Retrieval was held fixed at hybrid (dense + BM25, RRF fusion, no reranker).

| chunking | chunks | mean tokens | > embedder window | recall@5 | MRR | recall @1000 tok | recall @2000 tok | prompt tokens | $ / 1k queries* | index time |
|---|---|---|---|---|---|---|---|---|---|---|
| **structure-500 (default)** | 244 | 456 | 99.6% | 0.833 | 0.608 | 0.560 | 0.780 | 2,838 | 0.39 | 5.8 s |
| structure-250 | 537 | 212 | 25.5% | 0.820 | 0.602 | **0.807** | 0.887 | **1,626** | **0.27** | 11.2 s |
| structure-1000 | 116 | 942 | 100% | 0.787 | 0.586 | 0.447 | 0.560 | 5,224 | 0.63 | 2.4 s |
| fixed-250 | 441 | 248 | 46.3% | 0.833 | 0.625 | 0.760 | **0.913** | 1,770 | 0.28 | 8.6 s |
| fixed-500 | 221 | 493 | 99.1% | 0.820 | 0.563 | 0.513 | 0.773 | 3,012 | 0.40 | 4.3 s |
| fixed-500, no overlap | 179 | 491 | 98.9% | 0.753 | 0.554 | 0.540 | 0.713 | 3,010 | 0.40 | 3.4 s |
| sentence-250 | 444 | 228 | 31.3% | 0.760 | 0.596 | 0.707 | 0.847 | 1,675 | 0.27 | 9.1 s |
| sentence-500 | 219 | 475 | 99.5% | 0.820 | 0.614 | 0.600 | 0.787 | 2,915 | 0.39 | 4.2 s |
| semantic-500 | 419 | 208 | 38.4% | 0.793 | 0.625 | 0.693 | 0.820 | 2,206 | 0.32 | 18.7 s |

<sub>* Answer-generation call only. 95% CIs on recall@5 are about ±0.08.</sub>

1. **Recall@k on its own misleads.** Recall@5 is flat across strategies: none differs significantly from the default (smallest p = 0.18). But five 1,000-token chunks hold four times as much text as five 250-token ones, so bigger chunks get recall for free. When the context budget is fixed instead, small chunks win clearly (`structure-250` vs default at 1,000 tokens: 0.807 vs 0.560, p < 0.001). They put the evidence in front of the LLM using fewer tokens, which means lower cost, lower latency, and less irrelevant text for the model to be distracted by.
2. **Chunk size matters more than chunk method.** Structure-aware, fixed-window, and sentence-packing chunkers of the same size land within noise of each other.
3. **Overlap is probably worth keeping.** Removing it from 500-token windows drops dense-only recall@5 from 0.720 to 0.587 (p = 0.047). Hybrid recall falls from 0.820 to 0.753, but that is within noise (p = 0.22). Overlap costs only 24% more indexed tokens, and it protects evidence that sits on a chunk boundary.
4. **Semantic chunking didn't pay off here.**
   - Accuracy is no better than plain chunking.
   - Indexing is 3.2× slower and embeds 1.7× more tokens, because every sentence is embedded once to find split points.
   - It is the only strategy that split an evidence span past recovery (maximum achievable recall 0.987).
5. **Chunks that overflow the embedder's window are partly invisible.** 99.6% of the default chunks exceed MiniLM's 256-wordpiece window, so the embedder never sees their tails. Dense-only recall drops steadily with chunk size: 0.820 (fixed-250), 0.687 (structure-500), 0.613 (structure-1000). Gemini embeddings accept 2,048 tokens, so this mainly hurts the offline fallback, but nothing warned about it until it was measured.

## Retrieval and reranking

Chunking was held fixed at `structure-500`, the app default.

| retrieval | hit@1 | recall@5 (95% CI) | MRR (95% CI) | nDCG@10 | p50 latency | p vs hybrid (MRR / recall) |
|---|---|---|---|---|---|---|
| dense only | 0.427 | 0.687 (0.59–0.79) | 0.542 (0.44–0.64) | 0.561 | 2.9 ms | 0.10 / **0.001** |
| BM25 only | 0.453 | 0.760 (0.67–0.85) | 0.579 (0.49–0.68) | 0.629 | 0.9 ms | 0.49 / 0.19 |
| **hybrid (RRF)** | 0.467 | 0.833 (0.75–0.91) | 0.608 (0.52–0.70) | 0.661 | 4.0 ms | – |
| hybrid + keyword rerank | 0.520 | 0.880 (0.81–0.95) | 0.679 (0.60–0.76) | 0.737 | 6.8 ms | 0.053 / 0.30 |
| hybrid + MMR | 0.427 | 0.673 (0.56–0.78) | 0.543 (0.45–0.63) | 0.554 | 7.7 ms | 0.13 / **0.002** |
| hybrid + sentence max-sim | **0.653** | **0.900** (0.83–0.96) | **0.754** (0.67–0.83) | **0.786** | 1,640 ms | **0.005** / 0.13 |
| hybrid + cross-encoder | not run: model download blocked | | | | | |

1. **Hybrid search works because the two retrievers fail differently.** On lexical questions BM25 gets 0.778 and dense gets 0.500. On cross-document questions dense gets 0.750 and BM25 0.625. Hybrid gets 1.000 on cross-document. The gap is largest when the user's wording differs from the document's.
2. **Keyword rerank: a cheap win on big chunks, neutral on small ones.** It adds 7 points of MRR at the default size, but on 250-token chunks it is neutral (0.827 vs 0.820). On `fixed-250` it is slightly negative (0.807 vs 0.833). Rerankers and chunk size interact, so they shouldn't be tuned independently.
3. **Sentence max-sim** scores a chunk by its single best-matching sentence. It is the best ranker here: hit@1 rises from 0.47 to 0.65. Its gain is biggest on large chunks (MRR 0.765 vs 0.586 on `structure-1000`, p < 0.001) and is not significant at 250 tokens (0.660 vs 0.602, p = 0.28). That is consistent with large chunks' embeddings being diluted or truncated. As implemented, it embeds every candidate sentence per query, costing 1.6 s on CPU. Precomputing sentence vectors at index time would cut that to milliseconds.
4. **MMR reduces redundancy, but redundancy wasn't the problem here.** Redundant overlapping chunks weren't what kept answers out of the top 5. And MMR reorders using the dense similarity alone, so it throws away the BM25 signal and falls back to dense-only quality.
5. **The cross-encoder fell back silently.** In the app, `RERANK_STRATEGY=cross-encoder` silently became keyword overlap when the model couldn't be downloaded. The trace now records `rerank_strategy_used`, and the harness refuses to report a cross-encoder number that was really keyword overlap.

## Speed and cost per query

- **Retrieval is cheap.** p50 is 3–8 ms for every strategy except sentence max-sim. Every chat response now includes a `usage` object with per-stage milliseconds, token counts, and dollar cost.
- **BM25 was being rebuilt on every query.** The index was rebuilt from scratch for each question: 22–25 ms per query at 52k words, growing linearly with corpus size. Caching it (`reranker._get_bm25`) cuts this to about 1 ms with identical results; this is covered by a test.
- **Projected cost of the default pipeline: about $0.48 per 1,000 queries.** The components:
  - Answer generation: $0.39 per 1,000 queries.
  - Query rewriting and follow-up question generation: about 23% more on top.
  - Query embeddings: negligible.

  Chunk size is the biggest cost lever after model choice: answer cost runs from $0.27 (250-token chunks) to $0.63 (1,000-token chunks) per 1,000 queries. Indexing the whole corpus costs about $0.02 regardless of strategy. Prices are in `app/core/pricing.py`; check them before quoting.
- **Live latency will be dominated by LLM calls.** The default pipeline makes three sequential Gemini calls per question (rewrite, answer, follow-ups). Retrieval is not the place to optimise. Dropping the follow-up call, or making it asynchronous, removes a full LLM round trip.

## Faithfulness: do answers stick to the source?

**Calibrating the judge.** Before trusting a faithfulness score, the harness runs the judge on answers whose faithfulness is already known:

| judge | gold evidence (want 1.0) | faithful paraphrase (want 1.0) | injected foreign claim caught | changed number caught |
|---|---|---|---|---|
| lexical (offline) | 0.993 | **0.423** | 0.867 | 0.800 |

The offline lexical judge catches most copied-in and altered claims. But it rates correct paraphrases as 58% unfaithful, so it is unusable for LLM-written answers. Use `--judge llm`, which does claim decomposition plus verification, for Gemini answers, and calibrate it the same way (`--calibrate-judge llm`).

**Offline answer quality.**

- **Faithful by construction.** The extractive fallback scores 0.975–0.997 faithfulness for every configuration, as expected for text copied from the sources.
- **Often unhelpful.** It covers only 38–55% of the reference answer's key terms. In "medium" mode it shows the first sentences of the top document, which for single-page files is often the document header.
- **Never abstains.** It declined 0 of 6 unanswerable questions: it always shows passages even when they don't answer the question. These are worth fixing before relying on offline mode.

## Recommendations

1. **Set `CHUNK_SIZE=250` and `CHUNK_OVERLAP=50`** (structure-aware). Recall@5 is the same, with 44% more recall at a 1,000-token budget, 31% lower answer cost, and the chunks fit the local embedder. This report doesn't change the default; that is a product decision, and it should be confirmed with Gemini embeddings first.
2. **Keep hybrid RRF with overlap.** Drop MMR; skip semantic chunking.
3. **Measure the cross-encoder** on a machine with Hugging Face access before relying on it, and alert when `rerank_strategy_used` differs from what was configured.
4. **If ranking quality matters more than latency**, precompute sentence embeddings so sentence max-sim costs milliseconds instead of 1.6 s.
5. **Run the Gemini end-to-end pass:**
   `python -m evaluation.run --chunking structure-250 structure-500 --retrieval hybrid+keyword hybrid+rewrite+cross-encoder --generate --judge llm --calibrate-judge llm`
   This measures real faithfulness, correctness, abstention, tokens and latency.

## Limitations

- 75 answerable questions give CIs of about ±0.08, so small differences are noise. The p-values are what separate signal from noise.
- One corpus domain: technical, English, mostly single-page text files.
- The questions were written by an LLM (Claude) from the documents; the evidence spans are verified verbatim. LLM-written questions can lean towards the documents' own wording, which favours BM25. 45 of the 75 answerable questions were deliberately written as paraphrases to counter this.
- Embeddings are local MiniLM, not Gemini; absolute numbers will differ with Gemini embeddings.
- Projected costs estimate tokens as characters ÷ 4 and assume an output length; measured usage from a `--generate` run with a key replaces both.
