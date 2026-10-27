# Evaluation results: full-offline-refactor

- Date: 2026-10-27T11:17:24  |  git: `47bcba1`  |  embedder: all-MiniLM-L6-v2 (onnx)  |  LLM for cost projection: gemini-2.5-flash
- Questions: 81 (75 answerable)  |  retrieval depth scored: top-10  |  chunks sent to LLM: top-6

**Skipped configurations**

- `hybrid+cross-encoder`: cross-encoder model unavailable (sentence-transformers not installed or model download blocked); the app would fall back to keyword reranking, reported as rerank_strategy_used and in /api/status
- `hybrid+rewrite+cross-encoder`: query rewriting needs an LLM (GEMINI_API_KEY or an OpenAI-compatible server)

## Chunking strategies (retrieval = `hybrid-rrf`)

| chunking | chunks | mean tok | % truncated by embedder | max recall | hit@5 | recall@5 | MRR | nDCG@10 | P@5 | recall@1000tok | recall@2000tok | prompt tok | $/1k q (answer call) | index s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| structure-300 | 491 | 197 | 29.3 | 1.000 | 0.893 | 0.873 | 0.741 | 0.774 | 0.203 | 0.853 | 0.933 | 1739 | 1.150 | 14.3 |
| structure-500 | 322 | 296 | 58.4 | 1.000 | 0.893 | 0.880 | 0.675 | 0.731 | 0.197 | 0.747 | 0.880 | 2353 | 1.334 | 9.2 |
| structure-250 | 591 | 170 | 11.5 | 1.000 | 0.880 | 0.860 | 0.771 | 0.792 | 0.203 | 0.880 | 0.933 | 1559 | 1.096 | 15.4 |
| structure-1000 | 189 | 478 | 91.5 | 1.000 | 0.907 | 0.907 | 0.670 | 0.738 | 0.197 | 0.580 | 0.793 | 3630 | 1.717 | 5.1 |
| fixed-250 | 441 | 248 | 46.3 | 1.000 | 0.920 | 0.913 | 0.712 | 0.744 | 0.229 | 0.900 | 0.947 | 1941 | 1.210 | 10.6 |
| fixed-500 | 221 | 493 | 99.1 | 1.000 | 0.853 | 0.847 | 0.685 | 0.684 | 0.208 | 0.707 | 0.813 | 3423 | 1.655 | 6.0 |
| fixed-500-no-overlap | 179 | 491 | 98.9 | 1.000 | 0.800 | 0.800 | 0.596 | 0.668 | 0.168 | 0.533 | 0.760 | 3425 | 1.655 | 4.3 |
| sentence-250 | 444 | 228 | 31.3 | 1.000 | 0.893 | 0.880 | 0.670 | 0.710 | 0.219 | 0.840 | 0.947 | 1826 | 1.175 | 11.2 |
| sentence-500 | 219 | 475 | 99.5 | 1.000 | 0.867 | 0.860 | 0.697 | 0.723 | 0.216 | 0.727 | 0.807 | 3327 | 1.626 | 5.7 |
| semantic-500 | 419 | 208 | 38.4 | 0.987 | 0.840 | 0.833 | 0.677 | 0.737 | 0.173 | 0.727 | 0.820 | 2378 | 1.341 | 27.6 |

## Retrieval and reranking strategies (chunking = `structure-300`)

| retrieval | hit@1 | hit@5 | recall@5 (95% CI) | MRR (95% CI) | nDCG@10 | P@5 | p50 ms | p95 ms | rerank p50 ms | bm25 p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| dense-only | 0.560 | 0.853 | 0.847 (0.76–0.92) | 0.690 (0.61–0.77) | 0.735 | 0.200 | 6.0 | 9.5 | 1.6 | – |
| bm25-only | 0.587 | 0.880 | 0.867 (0.79–0.93) | 0.712 (0.63–0.80) | 0.750 | 0.200 | 1.9 | 2.8 | 1.6 | 0.2 |
| hybrid-rrf | 0.600 | 0.893 | 0.873 (0.80–0.94) | 0.741 (0.66–0.82) | 0.774 | 0.203 | 7.2 | 14.4 | 1.9 | 0.3 |
| hybrid+keyword | 0.600 | 0.867 | 0.853 (0.77–0.93) | 0.719 (0.64–0.80) | 0.750 | 0.197 | 6.1 | 9.5 | 1.6 | 0.2 |
| hybrid+mmr | 0.560 | 0.867 | 0.853 (0.77–0.92) | 0.686 (0.61–0.77) | 0.726 | 0.195 | 8.4 | 11.5 | 4.3 | 0.2 |
| hybrid+sentence-maxsim | 0.533 | 0.893 | 0.887 (0.81–0.95) | 0.678 (0.59–0.76) | 0.734 | 0.205 | 1004.6 | 1424.3 | 999.5 | 0.4 |

### recall@5 by question type

| retrieval | cross_doc | lexical | multi_span | paraphrase |
|---|---|---|---|---|
| dense-only | 1.000 | 0.722 | 0.688 | 0.911 |
| bm25-only | 1.000 | 0.889 | 0.562 | 0.900 |
| hybrid-rrf | 1.000 | 0.778 | 0.875 | 0.900 |
| hybrid+keyword | 1.000 | 0.833 | 0.812 | 0.856 |
| hybrid+mmr | 1.000 | 0.722 | 0.812 | 0.900 |
| hybrid+sentence-maxsim | 1.000 | 0.889 | 0.688 | 0.911 |

## Grid: mrr

| chunking \ retrieval | dense-only | bm25-only | hybrid-rrf | hybrid+keyword | hybrid+mmr | hybrid+sentence-maxsim |
|---|---|---|---|---|---|---|
| structure-300 | 0.690 | 0.712 | 0.741 | 0.719 | 0.686 | 0.678 |
| structure-500 | 0.585 | 0.740 | 0.675 | 0.749 | 0.586 | 0.719 |
| structure-250 | 0.685 | 0.742 | 0.771 | 0.756 | 0.679 | 0.665 |
| structure-1000 | 0.566 | 0.781 | 0.670 | 0.765 | 0.560 | 0.767 |
| fixed-250 | 0.632 | 0.691 | 0.712 | 0.710 | 0.628 | 0.634 |
| fixed-500 | 0.571 | 0.728 | 0.685 | 0.737 | 0.559 | 0.730 |
| fixed-500-no-overlap | 0.492 | 0.749 | 0.596 | 0.712 | 0.486 | 0.707 |
| sentence-250 | 0.621 | 0.709 | 0.670 | 0.691 | 0.616 | 0.617 |
| sentence-500 | 0.629 | 0.760 | 0.697 | 0.770 | 0.618 | 0.639 |
| semantic-500 | 0.526 | 0.756 | 0.677 | 0.758 | 0.500 | 0.665 |

## Grid: evidence_recall@5

| chunking \ retrieval | dense-only | bm25-only | hybrid-rrf | hybrid+keyword | hybrid+mmr | hybrid+sentence-maxsim |
|---|---|---|---|---|---|---|
| structure-300 | 0.847 | 0.867 | 0.873 | 0.853 | 0.853 | 0.887 |
| structure-500 | 0.773 | 0.880 | 0.880 | 0.880 | 0.713 | 0.887 |
| structure-250 | 0.840 | 0.887 | 0.860 | 0.873 | 0.847 | 0.867 |
| structure-1000 | 0.727 | 0.893 | 0.907 | 0.867 | 0.673 | 0.933 |
| fixed-250 | 0.813 | 0.873 | 0.913 | 0.887 | 0.813 | 0.853 |
| fixed-500 | 0.760 | 0.900 | 0.847 | 0.900 | 0.700 | 0.913 |
| fixed-500-no-overlap | 0.627 | 0.913 | 0.800 | 0.880 | 0.587 | 0.880 |
| sentence-250 | 0.793 | 0.840 | 0.880 | 0.867 | 0.733 | 0.833 |
| sentence-500 | 0.767 | 0.887 | 0.860 | 0.920 | 0.773 | 0.860 |
| semantic-500 | 0.680 | 0.887 | 0.833 | 0.887 | 0.693 | 0.887 |

## End-to-end generation

| chunking | retrieval | faithfulness | correctness | ref-token recall | false abstention | abstains on unanswerable | LLM calls/q | in tok | out tok | $/1k q (measured) | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| structure-300 | dense-only | 0.994 | – | 0.356 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 7 | 11 |
| structure-300 | bm25-only | 0.997 | – | 0.371 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 4 |
| structure-300 | hybrid-rrf | 0.997 | – | 0.359 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 9 | 18 |
| structure-300 | hybrid+keyword | 0.997 | – | 0.371 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 8 | 11 |
| structure-300 | hybrid+mmr | 0.991 | – | 0.370 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 10 | 14 |
| structure-300 | hybrid+sentence-maxsim | 0.997 | – | 0.357 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 993 | 1355 |
| structure-500 | dense-only | 0.994 | – | 0.331 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 10 | 13 |
| structure-500 | bm25-only | 0.994 | – | 0.350 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 4 | 6 |
| structure-500 | hybrid-rrf | 0.994 | – | 0.330 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 8 | 12 |
| structure-500 | hybrid+keyword | 0.997 | – | 0.346 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 9 | 12 |
| structure-500 | hybrid+mmr | 0.991 | – | 0.337 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 10 | 14 |
| structure-500 | hybrid+sentence-maxsim | 0.997 | – | 0.335 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1436 | 1927 |
| structure-250 | dense-only | 0.990 | – | 0.350 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 7 | 9 |
| structure-250 | bm25-only | 0.996 | – | 0.382 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 3 |
| structure-250 | hybrid-rrf | 0.993 | – | 0.366 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 7 | 12 |
| structure-250 | hybrid+keyword | 0.991 | – | 0.372 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 8 | 12 |
| structure-250 | hybrid+mmr | 0.988 | – | 0.369 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 10 | 14 |
| structure-250 | hybrid+sentence-maxsim | 0.993 | – | 0.354 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 791 | 1053 |
| structure-1000 | dense-only | 0.997 | – | 0.325 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 11 | 15 |
| structure-1000 | bm25-only | 0.997 | – | 0.337 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 6 | 10 |
| structure-1000 | hybrid-rrf | 0.994 | – | 0.327 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 11 | 15 |
| structure-1000 | hybrid+keyword | 0.997 | – | 0.329 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 12 | 15 |
| structure-1000 | hybrid+mmr | 0.997 | – | 0.330 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 12 | 15 |
| structure-1000 | hybrid+sentence-maxsim | 0.997 | – | 0.337 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 2386 | 3121 |
| fixed-250 | dense-only | 0.992 | – | 0.350 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 7 | 9 |
| fixed-250 | bm25-only | 0.990 | – | 0.335 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 4 |
| fixed-250 | hybrid-rrf | 0.988 | – | 0.366 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 9 | 15 |
| fixed-250 | hybrid+keyword | 0.993 | – | 0.358 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 8 | 12 |
| fixed-250 | hybrid+mmr | 0.995 | – | 0.349 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 11 | 19 |
| fixed-250 | hybrid+sentence-maxsim | 0.994 | – | 0.331 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1180 | 1560 |
| fixed-500 | dense-only | 0.996 | – | 0.337 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 11 | 14 |
| fixed-500 | bm25-only | 0.994 | – | 0.334 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 6 | 8 |
| fixed-500 | hybrid-rrf | 0.993 | – | 0.346 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 11 | 14 |
| fixed-500 | hybrid+keyword | 1.000 | – | 0.340 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 11 | 14 |
| fixed-500 | hybrid+mmr | 0.996 | – | 0.321 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 12 | 17 |
| fixed-500 | hybrid+sentence-maxsim | 0.997 | – | 0.327 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 2251 | 2684 |
| fixed-500-no-overlap | dense-only | 0.995 | – | 0.300 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 10 | 11 |
| fixed-500-no-overlap | bm25-only | 0.994 | – | 0.332 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 6 | 7 |
| fixed-500-no-overlap | hybrid-rrf | 0.991 | – | 0.314 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 11 | 14 |
| fixed-500-no-overlap | hybrid+keyword | 0.997 | – | 0.337 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 12 | 16 |
| fixed-500-no-overlap | hybrid+mmr | 0.995 | – | 0.306 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 12 | 19 |
| fixed-500-no-overlap | hybrid+sentence-maxsim | 0.997 | – | 0.341 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 2349 | 2750 |
| sentence-250 | dense-only | 0.990 | – | 0.339 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 8 | 11 |
| sentence-250 | bm25-only | 0.982 | – | 0.371 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 3 |
| sentence-250 | hybrid-rrf | 0.990 | – | 0.345 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 9 | 12 |
| sentence-250 | hybrid+keyword | 0.985 | – | 0.350 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 10 | 14 |
| sentence-250 | hybrid+mmr | 0.990 | – | 0.330 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 12 | 17 |
| sentence-250 | hybrid+sentence-maxsim | 0.991 | – | 0.339 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1151 | 1454 |
| sentence-500 | dense-only | 0.982 | – | 0.347 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 12 | 19 |
| sentence-500 | bm25-only | 0.984 | – | 0.342 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 6 | 8 |
| sentence-500 | hybrid-rrf | 0.979 | – | 0.357 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 11 | 15 |
| sentence-500 | hybrid+keyword | 0.982 | – | 0.356 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 12 | 14 |
| sentence-500 | hybrid+mmr | 0.971 | – | 0.350 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 12 | 17 |
| sentence-500 | hybrid+sentence-maxsim | 0.988 | – | 0.327 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 2031 | 2348 |
| semantic-500 | dense-only | 0.994 | – | 0.332 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 9 | 12 |
| semantic-500 | bm25-only | 0.983 | – | 0.349 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 4 | 5 |
| semantic-500 | hybrid-rrf | 1.000 | – | 0.353 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 9 | 14 |
| semantic-500 | hybrid+keyword | 0.990 | – | 0.357 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 9 | 13 |
| semantic-500 | hybrid+mmr | 0.993 | – | 0.329 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 10 | 13 |
| semantic-500 | hybrid+sentence-maxsim | 0.997 | – | 0.333 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1342 | 1857 |

## Faithfulness judge calibration

| judge | faithful extractive (want 1.0) | faithful paraphrase (want 1.0) | injected claim caught | changed numbers caught |
|---|---|---|---|---|
| lexical | 0.993 | 0.423 | 0.867 | 0.800 |

## Paired bootstrap p-values vs `structure-300 / hybrid-rrf`

| config | MRR p | recall@5 p |
|---|---|---|
| structure-300 / dense-only | 0.107 | 0.441 |
| structure-300 / bm25-only | 0.433 | 0.816 |
| structure-300 / hybrid+keyword | 0.542 | 0.519 |
| structure-300 / hybrid+mmr | 0.087 | 0.541 |
| structure-300 / hybrid+sentence-maxsim | 0.169 | 0.741 |
| structure-500 / dense-only | 0.000 | 0.050 |
| structure-500 / bm25-only | 0.988 | 0.824 |
| structure-500 / hybrid-rrf | 0.072 | 0.822 |
| structure-500 / hybrid+keyword | 0.842 | 0.827 |
| structure-500 / hybrid+mmr | 0.001 | 0.002 |
| structure-500 / hybrid+sentence-maxsim | 0.623 | 0.750 |
| structure-250 / dense-only | 0.122 | 0.392 |
| structure-250 / bm25-only | 0.981 | 0.690 |
| structure-250 / hybrid-rrf | 0.245 | 0.672 |
| structure-250 / hybrid+keyword | 0.692 | 1.000 |
| structure-250 / hybrid+mmr | 0.092 | 0.540 |
| structure-250 / hybrid+sentence-maxsim | 0.108 | 0.869 |
| structure-1000 / dense-only | 0.002 | 0.011 |
| structure-1000 / bm25-only | 0.315 | 0.623 |
| structure-1000 / hybrid-rrf | 0.144 | 0.479 |
| structure-1000 / hybrid+keyword | 0.548 | 0.861 |
| structure-1000 / hybrid+mmr | 0.002 | 0.001 |
| structure-1000 / hybrid+sentence-maxsim | 0.563 | 0.123 |
| fixed-250 / dense-only | 0.022 | 0.193 |
| fixed-250 / bm25-only | 0.259 | 1.000 |
| fixed-250 / hybrid-rrf | 0.479 | 0.303 |
| fixed-250 / hybrid+keyword | 0.488 | 0.733 |
| fixed-250 / hybrid+mmr | 0.031 | 0.247 |
| fixed-250 / hybrid+sentence-maxsim | 0.037 | 0.654 |
| fixed-500 / dense-only | 0.000 | 0.056 |
| fixed-500 / bm25-only | 0.776 | 0.461 |
| fixed-500 / hybrid-rrf | 0.146 | 0.592 |
| fixed-500 / hybrid+keyword | 0.927 | 0.470 |
| fixed-500 / hybrid+mmr | 0.000 | 0.005 |
| fixed-500 / hybrid+sentence-maxsim | 0.837 | 0.360 |
| fixed-500-no-overlap / dense-only | 0.000 | 0.000 |
| fixed-500-no-overlap / bm25-only | 0.840 | 0.246 |
| fixed-500-no-overlap / hybrid-rrf | 0.010 | 0.135 |
| fixed-500-no-overlap / hybrid+keyword | 0.571 | 0.846 |
| fixed-500-no-overlap / hybrid+mmr | 0.000 | 0.000 |
| fixed-500-no-overlap / hybrid+sentence-maxsim | 0.496 | 0.889 |
| sentence-250 / dense-only | 0.012 | 0.137 |
| sentence-250 / bm25-only | 0.483 | 0.448 |
| sentence-250 / hybrid-rrf | 0.095 | 0.873 |
| sentence-250 / hybrid+keyword | 0.301 | 0.845 |
| sentence-250 / hybrid+mmr | 0.013 | 0.007 |
| sentence-250 / hybrid+sentence-maxsim | 0.004 | 0.400 |
| sentence-500 / dense-only | 0.036 | 0.078 |
| sentence-500 / bm25-only | 0.643 | 0.731 |
| sentence-500 / hybrid-rrf | 0.313 | 0.781 |
| sentence-500 / hybrid+keyword | 0.527 | 0.232 |
| sentence-500 / hybrid+mmr | 0.019 | 0.083 |
| sentence-500 / hybrid+sentence-maxsim | 0.025 | 0.763 |
| semantic-500 / dense-only | 0.000 | 0.001 |
| semantic-500 / bm25-only | 0.724 | 0.737 |
| semantic-500 / hybrid-rrf | 0.225 | 0.394 |
| semantic-500 / hybrid+keyword | 0.690 | 0.699 |
| semantic-500 / hybrid+mmr | 0.000 | 0.003 |
| semantic-500 / hybrid+sentence-maxsim | 0.097 | 0.696 |
