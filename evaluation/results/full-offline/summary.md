# Evaluation results: full-offline

- Date: 2026-10-06T16:03:44  |  git: `2f100f4`  |  embedder: all-MiniLM-L6-v2 (onnx)  |  LLM for cost projection: gemini-2.0-flash
- Questions: 81 (75 answerable)  |  retrieval depth scored: top-10  |  chunks sent to LLM: top-5

**Skipped configurations**

- `hybrid+cross-encoder`: cross-encoder model unavailable (sentence-transformers not installed or model download blocked); the app would silently fall back to keyword overlap
- `hybrid+rewrite+cross-encoder`: query rewriting needs GEMINI_API_KEY

## Chunking strategies (retrieval = `hybrid-rrf`)

| chunking | chunks | mean tok | % truncated by embedder | max recall | hit@5 | recall@5 | MRR | nDCG@10 | P@5 | recall@1000tok | recall@2000tok | prompt tok | $/1k q (answer call) | index s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| structure-500 | 244 | 456 | 99.6 | 1.000 | 0.840 | 0.833 | 0.608 | 0.661 | 0.200 | 0.560 | 0.780 | 2838 | 0.387 | 5.8 |
| structure-250 | 537 | 212 | 25.5 | 1.000 | 0.827 | 0.820 | 0.602 | 0.636 | 0.197 | 0.807 | 0.887 | 1626 | 0.266 | 11.2 |
| structure-1000 | 116 | 942 | 100.0 | 1.000 | 0.787 | 0.787 | 0.586 | 0.632 | 0.205 | 0.447 | 0.560 | 5224 | 0.625 | 2.4 |
| fixed-250 | 441 | 248 | 46.3 | 1.000 | 0.840 | 0.833 | 0.625 | 0.663 | 0.192 | 0.760 | 0.913 | 1770 | 0.280 | 8.6 |
| fixed-500 | 221 | 493 | 99.1 | 1.000 | 0.827 | 0.820 | 0.563 | 0.621 | 0.205 | 0.513 | 0.773 | 3012 | 0.404 | 4.3 |
| fixed-500-no-overlap | 179 | 491 | 98.9 | 1.000 | 0.760 | 0.753 | 0.554 | 0.630 | 0.157 | 0.540 | 0.713 | 3010 | 0.404 | 3.4 |
| sentence-250 | 444 | 228 | 31.3 | 1.000 | 0.773 | 0.760 | 0.596 | 0.633 | 0.184 | 0.707 | 0.847 | 1675 | 0.270 | 9.1 |
| sentence-500 | 219 | 475 | 99.5 | 1.000 | 0.827 | 0.820 | 0.614 | 0.680 | 0.205 | 0.600 | 0.787 | 2915 | 0.394 | 4.2 |
| semantic-500 | 419 | 208 | 38.4 | 0.987 | 0.800 | 0.793 | 0.625 | 0.680 | 0.168 | 0.693 | 0.820 | 2206 | 0.324 | 18.7 |

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
| structure-1000 | 0.469 | 0.667 | 0.586 | 0.704 | 0.468 | 0.765 |
| fixed-250 | 0.662 | 0.496 | 0.625 | 0.653 | 0.667 | 0.659 |
| fixed-500 | 0.611 | 0.608 | 0.563 | 0.666 | 0.599 | 0.727 |
| fixed-500-no-overlap | 0.459 | 0.588 | 0.554 | 0.684 | 0.451 | 0.699 |
| sentence-250 | 0.594 | 0.504 | 0.596 | 0.677 | 0.569 | 0.623 |
| sentence-500 | 0.566 | 0.600 | 0.614 | 0.654 | 0.565 | 0.658 |
| semantic-500 | 0.525 | 0.581 | 0.625 | 0.687 | 0.494 | 0.668 |

## Grid: evidence_recall@5

| chunking \ retrieval | dense-only | bm25-only | hybrid-rrf | hybrid+keyword | hybrid+mmr | hybrid+sentence-maxsim |
|---|---|---|---|---|---|---|
| structure-500 | 0.687 | 0.760 | 0.833 | 0.880 | 0.673 | 0.900 |
| structure-250 | 0.740 | 0.600 | 0.820 | 0.827 | 0.740 | 0.820 |
| structure-1000 | 0.613 | 0.813 | 0.787 | 0.887 | 0.587 | 0.900 |
| fixed-250 | 0.820 | 0.640 | 0.833 | 0.807 | 0.833 | 0.873 |
| fixed-500 | 0.720 | 0.760 | 0.820 | 0.893 | 0.707 | 0.913 |
| fixed-500-no-overlap | 0.587 | 0.767 | 0.753 | 0.887 | 0.560 | 0.867 |
| sentence-250 | 0.767 | 0.607 | 0.760 | 0.747 | 0.693 | 0.833 |
| sentence-500 | 0.747 | 0.727 | 0.820 | 0.860 | 0.747 | 0.860 |
| semantic-500 | 0.680 | 0.680 | 0.793 | 0.853 | 0.627 | 0.860 |

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
| structure-1000 | dense-only | 0.988 | – | 0.397 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 8 |
| structure-1000 | bm25-only | 0.976 | – | 0.375 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1 | 1 |
| structure-1000 | hybrid-rrf | 0.975 | – | 0.424 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 4 | 6 |
| structure-1000 | hybrid+keyword | 0.983 | – | 0.379 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 8 | 9 |
| structure-1000 | hybrid+mmr | 0.984 | – | 0.402 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 7 | 8 |
| structure-1000 | hybrid+sentence-maxsim | 0.985 | – | 0.379 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3231 | 3898 |
| fixed-250 | dense-only | 0.994 | – | 0.535 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 5 |
| fixed-250 | bm25-only | 0.994 | – | 0.444 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1 | 2 |
| fixed-250 | hybrid-rrf | 0.997 | – | 0.490 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 4 | 7 |
| fixed-250 | hybrid+keyword | 0.994 | – | 0.497 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 6 | 7 |
| fixed-250 | hybrid+mmr | 0.993 | – | 0.538 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 7 | 9 |
| fixed-250 | hybrid+sentence-maxsim | 0.995 | – | 0.493 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 920 | 1143 |
| fixed-500 | dense-only | 0.991 | – | 0.491 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 4 |
| fixed-500 | bm25-only | 0.994 | – | 0.389 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1 | 1 |
| fixed-500 | hybrid-rrf | 0.989 | – | 0.444 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 4 | 5 |
| fixed-500 | hybrid+keyword | 0.991 | – | 0.423 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 6 | 8 |
| fixed-500 | hybrid+mmr | 0.990 | – | 0.498 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 7 | 9 |
| fixed-500 | hybrid+sentence-maxsim | 0.994 | – | 0.442 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1747 | 2107 |
| fixed-500-no-overlap | dense-only | 0.994 | – | 0.453 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 4 |
| fixed-500-no-overlap | bm25-only | 0.984 | – | 0.426 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1 | 1 |
| fixed-500-no-overlap | hybrid-rrf | 0.995 | – | 0.453 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 4 | 5 |
| fixed-500-no-overlap | hybrid+keyword | 0.991 | – | 0.447 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 6 | 7 |
| fixed-500-no-overlap | hybrid+mmr | 0.994 | – | 0.464 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 6 | 8 |
| fixed-500-no-overlap | hybrid+sentence-maxsim | 0.998 | – | 0.448 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1662 | 2115 |
| sentence-250 | dense-only | 0.996 | – | 0.499 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 4 |
| sentence-250 | bm25-only | 0.990 | – | 0.487 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1 | 2 |
| sentence-250 | hybrid-rrf | 0.993 | – | 0.505 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 5 | 6 |
| sentence-250 | hybrid+keyword | 0.995 | – | 0.528 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 6 | 7 |
| sentence-250 | hybrid+mmr | 0.995 | – | 0.503 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 7 | 9 |
| sentence-250 | hybrid+sentence-maxsim | 0.999 | – | 0.493 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 903 | 1110 |
| sentence-500 | dense-only | 0.993 | – | 0.515 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 5 |
| sentence-500 | bm25-only | 0.987 | – | 0.426 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1 | 1 |
| sentence-500 | hybrid-rrf | 0.988 | – | 0.480 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 4 | 5 |
| sentence-500 | hybrid+keyword | 0.990 | – | 0.434 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 6 | 8 |
| sentence-500 | hybrid+mmr | 0.991 | – | 0.530 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 7 | 9 |
| sentence-500 | hybrid+sentence-maxsim | 0.996 | – | 0.489 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1522 | 1862 |
| semantic-500 | dense-only | 0.995 | – | 0.483 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 3 | 6 |
| semantic-500 | bm25-only | 0.997 | – | 0.450 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1 | 2 |
| semantic-500 | hybrid-rrf | 0.997 | – | 0.449 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 4 | 6 |
| semantic-500 | hybrid+keyword | 0.995 | – | 0.487 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 6 | 8 |
| semantic-500 | hybrid+mmr | 0.992 | – | 0.494 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 7 | 10 |
| semantic-500 | hybrid+sentence-maxsim | 0.995 | – | 0.440 | 0.000 | 0.000 | 0.0 | 0 | 0 | 0.000 | 1162 | 1436 |

## Faithfulness judge calibration

| judge | faithful extractive (want 1.0) | faithful paraphrase (want 1.0) | injected claim caught | changed numbers caught |
|---|---|---|---|---|
| lexical | 0.993 | 0.423 | 0.867 | 0.800 |

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
| structure-1000 / dense-only | 0.021 | 0.000 |
| structure-1000 / bm25-only | 0.225 | 0.714 |
| structure-1000 / hybrid-rrf | 0.715 | 0.472 |
| structure-1000 / hybrid+keyword | 0.047 | 0.288 |
| structure-1000 / hybrid+mmr | 0.022 | 0.000 |
| structure-1000 / hybrid+sentence-maxsim | 0.004 | 0.137 |
| fixed-250 / dense-only | 0.328 | 0.804 |
| fixed-250 / bm25-only | 0.009 | 0.001 |
| fixed-250 / hybrid-rrf | 0.688 | 1.000 |
| fixed-250 / hybrid+keyword | 0.319 | 0.636 |
| fixed-250 / hybrid+mmr | 0.286 | 1.000 |
| fixed-250 / hybrid+sentence-maxsim | 0.389 | 0.392 |
| fixed-500 / dense-only | 0.962 | 0.050 |
| fixed-500 / bm25-only | 0.999 | 0.211 |
| fixed-500 / hybrid-rrf | 0.314 | 0.787 |
| fixed-500 / hybrid+keyword | 0.176 | 0.218 |
| fixed-500 / hybrid+mmr | 0.879 | 0.056 |
| fixed-500 / hybrid+sentence-maxsim | 0.033 | 0.101 |
| fixed-500-no-overlap / dense-only | 0.012 | 0.000 |
| fixed-500-no-overlap / bm25-only | 0.680 | 0.255 |
| fixed-500-no-overlap / hybrid-rrf | 0.279 | 0.182 |
| fixed-500-no-overlap / hybrid+keyword | 0.073 | 0.233 |
| fixed-500-no-overlap / hybrid+mmr | 0.010 | 0.000 |
| fixed-500-no-overlap / hybrid+sentence-maxsim | 0.090 | 0.512 |
| sentence-250 / dense-only | 0.804 | 0.260 |
| sentence-250 / bm25-only | 0.026 | 0.000 |
| sentence-250 / hybrid-rrf | 0.810 | 0.183 |
| sentence-250 / hybrid+keyword | 0.166 | 0.099 |
| sentence-250 / hybrid+mmr | 0.517 | 0.028 |
| sentence-250 / hybrid+sentence-maxsim | 0.798 | 1.000 |
| sentence-500 / dense-only | 0.449 | 0.120 |
| sentence-500 / bm25-only | 0.883 | 0.065 |
| sentence-500 / hybrid-rrf | 0.862 | 0.784 |
| sentence-500 / hybrid+keyword | 0.309 | 0.558 |
| sentence-500 / hybrid+mmr | 0.446 | 0.137 |
| sentence-500 / hybrid+sentence-maxsim | 0.356 | 0.605 |
| semantic-500 / dense-only | 0.161 | 0.014 |
| semantic-500 / bm25-only | 0.607 | 0.010 |
| semantic-500 / hybrid-rrf | 0.766 | 0.506 |
| semantic-500 / hybrid+keyword | 0.106 | 0.709 |
| semantic-500 / hybrid+mmr | 0.060 | 0.001 |
| semantic-500 / hybrid+sentence-maxsim | 0.271 | 0.601 |
