# Evaluation results: full-offline

- Date: 2026-10-06T15:22:56  |  git: `821bff1`  |  embedder: all-MiniLM-L6-v2 (onnx)  |  LLM for cost projection: gemini-2.0-flash
- Questions: 81 (75 answerable)  |  retrieval depth scored: top-10  |  chunks sent to LLM: top-5

**Skipped configurations**

- `hybrid+cross-encoder`: cross-encoder model unavailable (sentence-transformers not installed or model download blocked); the app would silently fall back to keyword overlap
- `hybrid+rewrite+cross-encoder`: query rewriting needs GEMINI_API_KEY

## Chunking strategies (retrieval = `hybrid-rrf`)

| chunking | chunks | mean tok | % truncated by embedder | max recall | hit@5 | recall@5 | MRR | nDCG@10 | P@5 | recall@1000tok | recall@2000tok | prompt tok | $/1k q (answer call) | index s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| structure-500 | 244 | 456 | 99.6 | 1.000 | 0.840 | 0.833 | 0.608 | 0.661 | 0.200 | 0.560 | 0.780 | 2838 | 0.387 | 5.8 |
| structure-250 | 537 | 212 | 25.5 | 1.000 | 0.827 | 0.820 | 0.602 | 0.636 | 0.197 | 0.807 | 0.887 | 1626 | 0.266 | 11.2 |

## Retrieval and reranking strategies (chunking = `structure-500`)

| retrieval | hit@1 | hit@5 | recall@5 (95% CI) | MRR (95% CI) | nDCG@10 | P@5 | p50 ms | p95 ms | rerank p50 ms | bm25 p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| dense-only | 0.427 | 0.693 | 0.687 (0.59–0.79) | 0.542 (0.44–0.64) | 0.561 | 0.160 | 2.9 | 4.0 | 0.0 | – |
| bm25-only | 0.453 | 0.773 | 0.760 (0.67–0.85) | 0.579 (0.49–0.68) | 0.629 | 0.192 | 0.9 | 1.4 | 0.0 | 0.8 |
| hybrid-rrf | 0.467 | 0.840 | 0.833 (0.75–0.91) | 0.608 (0.52–0.70) | 0.661 | 0.200 | 4.0 | 6.5 | 0.0 | 1.0 |
| hybrid+keyword | 0.520 | 0.893 | 0.880 (0.81–0.95) | 0.679 (0.60–0.76) | 0.737 | 0.229 | 6.8 | 9.3 | 2.0 | 1.1 |
| hybrid+mmr | 0.427 | 0.680 | 0.673 (0.56–0.78) | 0.543 (0.45–0.63) | 0.554 | 0.160 | 7.7 | 12.3 | 3.3 | 1.1 |
| hybrid+sentence-maxsim | 0.653 | 0.907 | 0.900 (0.83–0.96) | 0.754 (0.67–0.83) | 0.786 | 0.237 | 1639.6 | 1955.3 | 1632.7 | 1.2 |

### recall@5 by question type

| retrieval | cross_doc | lexical | multi_span | paraphrase |
|---|---|---|---|---|
| dense-only | 0.750 | 0.500 | 0.562 | 0.778 |
| bm25-only | 0.625 | 0.778 | 0.562 | 0.800 |
| hybrid-rrf | 1.000 | 0.778 | 0.688 | 0.867 |
| hybrid+keyword | 0.875 | 0.833 | 0.812 | 0.911 |
| hybrid+mmr | 0.750 | 0.500 | 0.688 | 0.733 |
| hybrid+sentence-maxsim | 1.000 | 0.889 | 0.812 | 0.911 |

## Grid: mrr

| chunking \ retrieval | dense-only | bm25-only | hybrid-rrf | hybrid+keyword | hybrid+mmr | hybrid+sentence-maxsim |
|---|---|---|---|---|---|---|
| structure-500 | 0.542 | 0.579 | 0.608 | 0.679 | 0.543 | 0.754 |
| structure-250 | 0.629 | 0.500 | 0.602 | 0.686 | 0.622 | 0.660 |

## Grid: evidence_recall@5

| chunking \ retrieval | dense-only | bm25-only | hybrid-rrf | hybrid+keyword | hybrid+mmr | hybrid+sentence-maxsim |
|---|---|---|---|---|---|---|
| structure-500 | 0.687 | 0.760 | 0.833 | 0.880 | 0.673 | 0.900 |
| structure-250 | 0.740 | 0.600 | 0.820 | 0.827 | 0.740 | 0.820 |

## End-to-end generation

| chunking | retrieval | faithfulness | correctness | ref-token recall | false abstention | abstains on unanswerable | LLM calls/q | in tok | out tok | $/1k q (measured) | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| structure-500 | dense-only | 0.985 | – | 0.455 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 4 |
| structure-500 | bm25-only | 0.989 | – | 0.436 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1 | 2 |
| structure-500 | hybrid-rrf | 0.987 | – | 0.444 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 4 | 6 |
| structure-500 | hybrid+keyword | 0.986 | – | 0.422 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 7 | 10 |
| structure-500 | hybrid+mmr | 0.986 | – | 0.475 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 8 | 12 |
| structure-500 | hybrid+sentence-maxsim | 0.995 | – | 0.439 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1635 | 1964 |
| structure-250 | dense-only | 0.993 | – | 0.543 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 7 |
| structure-250 | bm25-only | 0.991 | – | 0.492 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 2 | 2 |
| structure-250 | hybrid-rrf | 0.988 | – | 0.505 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 5 | 8 |
| structure-250 | hybrid+keyword | 0.989 | – | 0.545 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 7 | 9 |
| structure-250 | hybrid+mmr | 0.996 | – | 0.558 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 9 | 13 |
| structure-250 | hybrid+sentence-maxsim | 0.988 | – | 0.546 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 823 | 1064 |

## Paired bootstrap p-values vs `structure-500 / hybrid-rrf`

| config | MRR p | recall@5 p |
|---|---|---|
| structure-500 / dense-only | 0.101 | 0.001 |
| structure-500 / bm25-only | 0.487 | 0.187 |
| structure-500 / hybrid+keyword | 0.053 | 0.295 |
| structure-500 / hybrid+mmr | 0.128 | 0.002 |
| structure-500 / hybrid+sentence-maxsim | 0.005 | 0.134 |
| structure-250 / dense-only | 0.725 | 0.111 |
| structure-250 / bm25-only | 0.036 | 0.000 |
| structure-250 / hybrid-rrf | 0.898 | 0.787 |
| structure-250 / hybrid+keyword | 0.081 | 0.887 |
| structure-250 / hybrid+mmr | 0.817 | 0.111 |
| structure-250 / hybrid+sentence-maxsim | 0.386 | 0.811 |
