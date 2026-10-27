"""
RAG evaluation harness.

Builds an in-memory index of the evaluation corpus for each chunking strategy,
runs every retrieval/reranking configuration over the golden question set, and
reports retrieval accuracy, latency per stage, and cost per query. With
--generate it also produces answers through the app's real generate_answer()
and scores faithfulness, correctness and abstention.

Examples:
    # Retrieval-only grid (offline, local MiniLM embeddings; ~10 min on 4 CPUs)
    python -m evaluation.run --suite full --name full-onnx

    # One config end-to-end with Gemini generation and an LLM judge
    python -m evaluation.run --chunking structure-500 --retrieval hybrid+cross-encoder \\
        --generate --judge llm --name gemini-e2e

    # Check how trustworthy the faithfulness judge is
    python -m evaluation.run --calibrate-judge lexical --suite none --name judge-cal

Nothing here touches the app's persisted vector store in data/vectorstore.
"""
import argparse
import datetime as dt
import json
import logging
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from app.core.config import settings
from app.core.pricing import cost_usd, estimate_tokens
from app.core.schemas import AnswerStyle
from app.core.telemetry import trace
from app.generation import llm as llm_module
from app.generation.answerer import format_context_passages
from app.generation.prompts import ANSWER_PROMPT_TEMPLATE, FOLLOWUP_INSTRUCTION, STYLE_INSTRUCTIONS, SYSTEM_PROMPT
from app.retrieval import reranker as reranker_module
from app.retrieval.chunker import chunk_document
from app.retrieval.retriever import RetrievalConfig, retrieve_relevant_chunks, top_k_for_style
from app.retrieval.vectorstore import VectorStore
from evaluation import metrics as M
from evaluation.backends import get_embedder
from evaluation.dataset import DEFAULT_DATASET, chunk_covers, load_corpus, load_dataset, validate
from evaluation.judges import calibrate, get_judge

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# chunk_size / chunk_overlap are in approximate tokens; build_index converts them to the
# chunker's characters (x4). The app default is CHUNK_SIZE=1200 / CHUNK_OVERLAP=200 characters.
CHARS_PER_TOKEN = 4
CHUNKING_CONFIGS: Dict[str, dict] = {
    "structure-300": dict(strategy="structure", chunk_size=300, chunk_overlap=50),  # app default
    "structure-500": dict(strategy="structure", chunk_size=500, chunk_overlap=100),  # previous default
    "structure-250": dict(strategy="structure", chunk_size=250, chunk_overlap=50),
    "structure-1000": dict(strategy="structure", chunk_size=1000, chunk_overlap=200),
    "fixed-250": dict(strategy="fixed", chunk_size=250, chunk_overlap=50),
    "fixed-500": dict(strategy="fixed", chunk_size=500, chunk_overlap=100),
    "fixed-500-no-overlap": dict(strategy="fixed", chunk_size=500, chunk_overlap=0),
    "sentence-250": dict(strategy="sentence", chunk_size=250, chunk_overlap=50),
    "sentence-500": dict(strategy="sentence", chunk_size=500, chunk_overlap=100),
    "semantic-500": dict(strategy="semantic", chunk_size=500, chunk_overlap=0),
}

RETRIEVAL_CONFIGS: Dict[str, RetrievalConfig] = {
    "dense-only": RetrievalConfig(use_bm25=False, use_query_rewrite=False, rerank_strategy="none"),
    "bm25-only": RetrievalConfig(use_dense=False, use_query_rewrite=False, rerank_strategy="none"),
    "hybrid-rrf": RetrievalConfig(use_query_rewrite=False, rerank_strategy="none"),
    "hybrid+keyword": RetrievalConfig(use_query_rewrite=False, rerank_strategy="keyword"),
    "hybrid+mmr": RetrievalConfig(use_query_rewrite=False, rerank_strategy="mmr"),
    "hybrid+sentence-maxsim": RetrievalConfig(use_query_rewrite=False, rerank_strategy="sentence-maxsim"),
    "hybrid+cross-encoder": RetrievalConfig(use_query_rewrite=False, rerank_strategy="cross-encoder"),
    # The app's full default pipeline (needs GEMINI_API_KEY and the cross-encoder model).
    "hybrid+rewrite+cross-encoder": RetrievalConfig(use_query_rewrite=True, rerank_strategy="cross-encoder"),
}

SUITES = {
    "chunking": (list(CHUNKING_CONFIGS), ["hybrid-rrf"]),
    "reranking": (["structure-300"], list(RETRIEVAL_CONFIGS)),
    "full": (list(CHUNKING_CONFIGS), list(RETRIEVAL_CONFIGS)),
    "none": ([], []),
}

BASELINE = ("structure-300", "hybrid-rrf")
EVAL_TOP_K = 10  # rank depth scored; the app passes the top top_k_for_style(style) passages to the LLM

# Model whose prices (app/core/pricing.py) are used to project per-query cost. It must be
# priced: the app's configured model may be an alias or a provider without a price entry.
COST_MODEL = "gemini-2.5-flash"

# Assumptions used only to *project* per-query cost when no LLM was called.
# Measured usage from a --generate run replaces them.
PROJECTED_OUTPUT_TOKENS = {AnswerStyle.CONCISE: 100, AnswerStyle.MEDIUM: 250, AnswerStyle.LONG: 900}
PROJECTED_REWRITE_TOKENS = (250, 40)    # (input, output) for the query-rewrite call
PROJECTED_FOLLOWUP_OUTPUT_TOKENS = 60   # follow-ups are generated in the answer call itself


def available(retrieval_name: str) -> (bool, str):
    rc = RETRIEVAL_CONFIGS[retrieval_name]
    if rc.use_query_rewrite and llm_module.llm is None:
        return False, "query rewriting needs an LLM (GEMINI_API_KEY or an OpenAI-compatible server)"
    if rc.rerank_strategy == "cross-encoder" and reranker_module.reranker._get_cross_encoder() is None:
        return False, ("cross-encoder model unavailable (sentence-transformers not installed or "
                       "model download blocked); the app would fall back to keyword reranking, reported "
                       "as rerank_strategy_used and in /api/status")
    return True, ""


def projected_cost(question: str, chunks: List[dict], style: AnswerStyle, full_pipeline: bool) -> Dict:
    """Cost of the LLM + embedding calls the app *would* make for this query with Gemini."""
    prompt = SYSTEM_PROMPT + ANSWER_PROMPT_TEMPLATE.format(
        chat_history="", context_passages=format_context_passages(chunks),
        style_instruction=STYLE_INSTRUCTIONS[style], question=question,
        followup_instruction=FOLLOWUP_INSTRUCTION if full_pipeline else "")
    in_tok, out_tok = estimate_tokens(prompt), PROJECTED_OUTPUT_TOKENS[style]
    n_query_embeddings = 1
    if full_pipeline:
        n_query_embeddings = 3  # standalone question + up to 2 rewrites, embedded in one call
        in_tok += PROJECTED_REWRITE_TOKENS[0]
        out_tok += PROJECTED_REWRITE_TOKENS[1] + PROJECTED_FOLLOWUP_OUTPUT_TOKENS
    llm = cost_usd(COST_MODEL, in_tok, out_tok)
    if llm is None:
        raise ValueError(f"No price for {COST_MODEL!r} in app/core/pricing.py; pass --cost-model with a priced model")
    emb = cost_usd(settings.EMBEDDING_MODEL, n_query_embeddings * estimate_tokens(question)) or 0.0
    return {"prompt_tokens": estimate_tokens(prompt), "input_tokens": in_tok, "output_tokens": out_tok,
            "cost_usd": llm + emb}


def build_index(corpus, chunk_cfg: dict, embedder, cache_bm25: bool = True):
    t0 = time.perf_counter()
    with trace() as chunk_trace:
        chunks = []
        for doc in corpus.docs.values():
            chunks.extend(chunk_document(
                doc, strategy=chunk_cfg["strategy"], embed_fn=embedder.embed_documents,
                chunk_size=chunk_cfg["chunk_size"] * CHARS_PER_TOKEN,
                chunk_overlap=chunk_cfg["chunk_overlap"] * CHARS_PER_TOKEN))
    chunk_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    store = VectorStore(embedder=embedder, persist=False)
    store.cache_bm25 = cache_bm25
    store.add_chunks(chunks)
    embed_s = time.perf_counter() - t1

    tok = [estimate_tokens(c["text"]) for c in chunks]
    stats = {
        "n_chunks": len(chunks),
        "mean_tokens": round(float(np.mean(tok)), 1),
        "p95_tokens": round(M.percentile(tok, 95), 1),
        "max_tokens": int(max(tok)),
        "indexed_tokens": int(sum(tok)),
        "chunking_seconds": round(chunk_s, 2),
        "embedding_seconds": round(embed_s, 2),
        "chunking_embedding_tokens": chunk_trace.summary()["embedding_tokens"],  # semantic chunking only
    }
    tokenizer = getattr(embedder, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.no_truncation()
        lengths = [sum(e.attention_mask) for e in tokenizer.encode_batch([c["text"] for c in chunks])]
        tokenizer.enable_truncation(max_length=embedder.max_length)
        stats["pct_truncated_by_embedder"] = round(100 * float(np.mean([n > embedder.max_length for n in lengths])), 1)
    # What indexing would cost with the app's API embedder (overlap and sentence embedding included).
    index_tokens = stats["indexed_tokens"] + stats["chunking_embedding_tokens"]
    stats["projected_index_cost_usd"] = cost_usd(settings.EMBEDDING_MODEL, index_tokens) or 0.0
    return store, chunks, stats


def run_retrieval(items, store, chunks, rc: RetrievalConfig, style, generate, judge, chunk_name, ret_name):
    per_query = []
    relevant_counts = {it.id: sum(any(chunk_covers(c, ev) for ev in it.evidence) for c in chunks)
                       for it in items if it.answerable}
    for item in items:
        with trace() as t:
            result = retrieve_relevant_chunks(item.question, top_k=EVAL_TOP_K, store=store, config=rc)
        scored = result.chunks
        ranked = [c for c, _ in scored]
        n_llm = top_k_for_style(style)
        usage = t.summary()
        row = {
            "chunking": chunk_name, "retrieval": ret_name, "id": item.id, "type": item.type,
            "retrieval_ms": usage["total_ms"], "stages_ms": usage["stages_ms"],
            "rerank_strategy_used": usage["annotations"].get("rerank_strategy_used"),
            "top_chunks": [c["chunk_id"] for c in ranked[:5]],
            "top_score": float(scored[0][1]) if scored else None,
            "mode": result.mode,
            "projected": projected_cost(item.question, ranked[:n_llm], style, full_pipeline=False),
            "projected_full": projected_cost(item.question, ranked[:n_llm], style, full_pipeline=True),
        }
        if item.answerable:
            row["metrics"] = M.retrieval_metrics(item, ranked, relevant_counts[item.id])
        if generate:
            from app.generation.answerer import generate_answer
            resp = generate_answer(item.question, style=style, store=store, retrieval_config=rc,
                                   generate_followups=generate == "full")
            by_id = {c["chunk_id"]: c for c in chunks}
            used = [by_id[s["chunk_id"]] for s in resp.sources_used if s.get("chunk_id") in by_id]
            row["answer"] = resp.answer
            row["generation_usage"] = resp.usage
            row["judge"] = judge.judge(item.question, resp.answer, used, item.reference_answer)
        per_query.append(row)
    return per_query


def summarize(rows, items, chunk_stats, max_recall) -> Dict:
    ans = [r for r in rows if "metrics" in r]
    keys = list(ans[0]["metrics"].keys()) if ans else []
    out = {"chunking": rows[0]["chunking"], "retrieval": rows[0]["retrieval"], "n_questions": len(rows),
           "n_answerable": len(ans), "chunk_stats": chunk_stats, "max_achievable_recall": max_recall}
    out["metrics"] = {k: round(M.mean([r["metrics"][k] for r in ans]), 4) for k in keys}
    for k in ("evidence_recall@5", "mrr", "hit@5"):
        lo, hi = M.bootstrap_ci([r["metrics"][k] for r in ans])
        out["metrics"][f"{k}_ci95"] = [round(lo, 4), round(hi, 4)]
    by_type = defaultdict(list)
    for r in ans:
        by_type[r["type"]].append(r["metrics"]["evidence_recall@5"])
    out["recall@5_by_type"] = {t: round(M.mean(v), 4) for t, v in sorted(by_type.items())}

    lat = [r["retrieval_ms"] for r in rows]
    stage_names = sorted({s for r in rows for s in r["stages_ms"]})
    out["latency_ms"] = {
        "retrieval_p50": round(M.percentile(lat, 50), 2),
        "retrieval_p95": round(M.percentile(lat, 95), 2),
        "stages_p50": {s: round(M.percentile([r["stages_ms"].get(s, 0.0) for r in rows], 50), 2) for s in stage_names},
    }
    out["projected_cost"] = {
        "model": COST_MODEL,
        "prompt_tokens_mean": round(M.mean([r["projected"]["prompt_tokens"] for r in rows]), 1),
        "answer_only_usd_per_1k_queries": round(1000 * M.mean([r["projected"]["cost_usd"] for r in rows]), 4),
        "full_pipeline_usd_per_1k_queries": round(1000 * M.mean([r["projected_full"]["cost_usd"] for r in rows]), 4),
    }
    out["rerank_strategy_used"] = dict(Counter(r["rerank_strategy_used"] for r in rows))

    gen = [r for r in rows if "judge" in r]
    if gen:
        answerable_ids = {i.id for i in items if i.answerable}
        g_ans = [r for r in gen if r["id"] in answerable_ids]
        g_un = [r for r in gen if r["id"] not in answerable_ids]
        usage = [r["generation_usage"] for r in gen]
        out["generation"] = {
            "faithfulness": round(M.mean([r["judge"]["faithfulness"] for r in g_ans]), 4),
            "correctness": round(M.mean([r["judge"].get("correctness") for r in g_ans]), 4)
            if any(r["judge"].get("correctness") is not None for r in g_ans) else None,
            "reference_token_recall": round(M.mean([r["judge"].get("reference_token_recall") for r in g_ans]), 4)
            if any(r["judge"].get("reference_token_recall") is not None for r in g_ans) else None,
            "false_abstention_rate": round(M.mean([float(r["judge"]["abstained"]) for r in g_ans]), 4),
            "unanswerable_abstention_rate": round(M.mean([float(r["judge"]["abstained"]) for r in g_un]), 4)
            if g_un else None,
            "end_to_end_ms_p50": round(M.percentile([u["total_ms"] for u in usage], 50), 1),
            "end_to_end_ms_p95": round(M.percentile([u["total_ms"] for u in usage], 95), 1),
            "llm_calls_per_query": round(M.mean([u["llm_calls"] for u in usage]), 2),
            "llm_input_tokens_mean": round(M.mean([u["llm_input_tokens"] for u in usage]), 1),
            "llm_output_tokens_mean": round(M.mean([u["llm_output_tokens"] for u in usage]), 1),
            "measured_usd_per_1k_queries": round(1000 * M.mean([u["estimated_cost_usd"] for u in usage]), 4)
            if all(u["estimated_cost_usd"] is not None for u in usage) else None,
        }
    return out


def significance(summaries, per_query) -> Dict:
    """Paired bootstrap p-values of MRR and recall@5 vs the baseline config."""
    by_cfg = defaultdict(dict)
    for r in per_query:
        if "metrics" in r:
            by_cfg[(r["chunking"], r["retrieval"])][r["id"]] = r["metrics"]
    base = by_cfg.get(BASELINE)
    if not base:
        return {}
    ids = sorted(base)
    out = {}
    for cfg, vals in by_cfg.items():
        if cfg == BASELINE or set(vals) != set(base):
            continue
        out[f"{cfg[0]} / {cfg[1]}"] = {
            m: M.paired_bootstrap_pvalue([vals[i][m] for i in ids], [base[i][m] for i in ids])
            for m in ("mrr", "evidence_recall@5")
        }
    return out


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_markdown(path: Path, env, summaries, skipped, calibration, pvals):
    L = [f"# Evaluation results: {env['name']}", "",
         f"- Date: {env['date']}  |  git: `{env['git_sha']}`  |  embedder: {env['embedder']}  |  "
         f"LLM for cost projection: {env['cost_model']}",
         f"- Questions: {env['n_questions']} ({env['n_answerable']} answerable)  |  retrieval depth scored: "
         f"top-{EVAL_TOP_K}  |  chunks sent to LLM: top-{env['llm_top_k']}", ""]
    if skipped:
        L += ["**Skipped configurations**", ""] + [f"- `{k}`: {v}" for k, v in skipped.items()] + [""]

    chunk_rows = [s for s in summaries if s["retrieval"] == BASELINE[1]]
    if chunk_rows:
        L += [f"## Chunking strategies (retrieval = `{BASELINE[1]}`)", "",
              "| chunking | chunks | mean tok | % truncated by embedder | max recall | hit@5 | recall@5 | MRR | "
              "nDCG@10 | P@5 | recall@1000tok | recall@2000tok | prompt tok | $/1k q (answer call) | index s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in chunk_rows:
            cs, m = s["chunk_stats"], s["metrics"]
            L.append(f"| {s['chunking']} | {cs['n_chunks']} | {cs['mean_tokens']:.0f} | "
                     f"{fmt(cs.get('pct_truncated_by_embedder'), 1)} | {fmt(s['max_achievable_recall'])} | "
                     f"{fmt(m['hit@5'])} | {fmt(m['evidence_recall@5'])} | {fmt(m['mrr'])} | {fmt(m['ndcg@10'])} | "
                     f"{fmt(m['precision@5'])} | {fmt(m['recall@1000tok'])} | {fmt(m['recall@2000tok'])} | "
                     f"{s['projected_cost']['prompt_tokens_mean']:.0f} | "
                     f"{fmt(s['projected_cost']['answer_only_usd_per_1k_queries'], 3)} | "
                     f"{cs['chunking_seconds'] + cs['embedding_seconds']:.1f} |")
        L.append("")

    ret_rows = [s for s in summaries if s["chunking"] == BASELINE[0]]
    if ret_rows:
        L += [f"## Retrieval and reranking strategies (chunking = `{BASELINE[0]}`)", "",
              "| retrieval | hit@1 | hit@5 | recall@5 (95% CI) | MRR (95% CI) | nDCG@10 | P@5 | "
              "p50 ms | p95 ms | rerank p50 ms | bm25 p50 ms |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in ret_rows:
            m, lat = s["metrics"], s["latency_ms"]
            ci_r, ci_m = m["evidence_recall@5_ci95"], m["mrr_ci95"]
            L.append(f"| {s['retrieval']} | {fmt(m['hit@1'])} | {fmt(m['hit@5'])} | "
                     f"{fmt(m['evidence_recall@5'])} ({fmt(ci_r[0], 2)}–{fmt(ci_r[1], 2)}) | "
                     f"{fmt(m['mrr'])} ({fmt(ci_m[0], 2)}–{fmt(ci_m[1], 2)}) | {fmt(m['ndcg@10'])} | "
                     f"{fmt(m['precision@5'])} | {lat['retrieval_p50']:.1f} | {lat['retrieval_p95']:.1f} | "
                     f"{fmt(lat['stages_p50'].get('rerank'), 1)} | {fmt(lat['stages_p50'].get('bm25'), 1)} |")
        L.append("")
        types = sorted({t for s in ret_rows for t in s["recall@5_by_type"]})
        L += ["### recall@5 by question type", "", "| retrieval | " + " | ".join(types) + " |",
              "|---|" + "---|" * len(types)]
        for s in ret_rows:
            L.append(f"| {s['retrieval']} | " + " | ".join(fmt(s["recall@5_by_type"].get(t)) for t in types) + " |")
        L.append("")

    chunk_names = list(dict.fromkeys(s["chunking"] for s in summaries))
    ret_names = list(dict.fromkeys(s["retrieval"] for s in summaries))
    if len(chunk_names) > 1 and len(ret_names) > 1:
        grid = {(s["chunking"], s["retrieval"]): s["metrics"] for s in summaries}
        for metric in ("mrr", "evidence_recall@5"):
            L += [f"## Grid: {metric}", "", "| chunking \\ retrieval | " + " | ".join(ret_names) + " |",
                  "|---|" + "---|" * len(ret_names)]
            for c in chunk_names:
                L.append(f"| {c} | " + " | ".join(fmt(grid.get((c, r), {}).get(metric)) for r in ret_names) + " |")
            L.append("")

    gen_rows = [s for s in summaries if "generation" in s]
    if gen_rows:
        L += ["## End-to-end generation", "",
              "| chunking | retrieval | faithfulness | correctness | ref-token recall | false abstention | "
              "abstains on unanswerable | LLM calls/q | in tok | out tok | $/1k q (measured) | p50 ms | p95 ms |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in gen_rows:
            g = s["generation"]
            L.append(f"| {s['chunking']} | {s['retrieval']} | {fmt(g['faithfulness'])} | {fmt(g['correctness'])} | "
                     f"{fmt(g['reference_token_recall'])} | {fmt(g['false_abstention_rate'])} | "
                     f"{fmt(g['unanswerable_abstention_rate'])} | {g['llm_calls_per_query']} | "
                     f"{g['llm_input_tokens_mean']:.0f} | {g['llm_output_tokens_mean']:.0f} | "
                     f"{fmt(g['measured_usd_per_1k_queries'], 3)} | {g['end_to_end_ms_p50']:.0f} | "
                     f"{g['end_to_end_ms_p95']:.0f} |")
        L.append("")

    if calibration:
        L += ["## Faithfulness judge calibration", "",
              "| judge | faithful extractive (want 1.0) | faithful paraphrase (want 1.0) | "
              "injected claim caught | changed numbers caught |", "|---|---|---|---|---|"]
        for c in calibration:
            L.append(f"| {c['judge']} | {fmt(c['faithful_extractive_score'])} | "
                     f"{fmt(c['faithful_abstractive_score'])} | {fmt(c['injected_claim_detection_rate'])} | "
                     f"{fmt(c['number_swap_detection_rate'])} |")
        L.append("")

    if pvals:
        L += [f"## Paired bootstrap p-values vs `{BASELINE[0]} / {BASELINE[1]}`", "",
              "| config | MRR p | recall@5 p |", "|---|---|---|"]
        for cfg, p in pvals.items():
            L.append(f"| {cfg} | {p['mrr']:.3f} | {p['evidence_recall@5']:.3f} |")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    global COST_MODEL
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=list(SUITES), default="full")
    parser.add_argument("--chunking", nargs="+", choices=list(CHUNKING_CONFIGS), help="overrides the suite's list")
    parser.add_argument("--retrieval", nargs="+", choices=list(RETRIEVAL_CONFIGS), help="overrides the suite's list")
    parser.add_argument("--embedder", choices=["onnx-minilm", "app"], default="onnx-minilm")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--style", choices=[s.value for s in AnswerStyle], default="medium")
    parser.add_argument("--generate", nargs="?", const="answer", choices=["answer", "full"],
                        help="also generate answers ('full' includes follow-up question generation)")
    parser.add_argument("--judge", choices=["lexical", "llm"], default="lexical")
    parser.add_argument("--calibrate-judge", nargs="*", choices=["lexical", "llm"], default=None)
    parser.add_argument("--cost-model", default=COST_MODEL,
                        help="priced model (see app/core/pricing.py) used to project per-query cost")
    parser.add_argument("--no-bm25-cache", action="store_true", help="rebuild BM25 per query (pre-fix behaviour)")
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--name", default=None, help="results sub-directory name")
    parser.add_argument("--resume", action="store_true",
                        help="keep configurations already saved under --name and run only the rest")
    args = parser.parse_args(argv)

    logging.getLogger("rag_system").setLevel(logging.WARNING)
    COST_MODEL = args.cost_model
    items = load_dataset(args.dataset)[: args.limit] if args.limit else load_dataset(args.dataset)
    corpus = load_corpus()
    problems = validate(items, corpus)
    if problems:
        print("Dataset problems:\n  " + "\n  ".join(problems))
        return 1

    chunk_names, ret_names = SUITES[args.suite]
    chunk_names = args.chunking or chunk_names
    ret_names = args.retrieval or ret_names
    skipped = {}
    runnable = []
    for r in ret_names:
        ok, why = available(r)
        (runnable.append(r) if ok else skipped.__setitem__(r, why))

    style = AnswerStyle(args.style)
    judge = get_judge(args.judge) if args.generate else None
    embedder = get_embedder(args.embedder)
    name = args.name or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, all_rows = [], []
    done = set()
    if args.resume and (out_dir / "summary.json").exists():
        summaries = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["runs"]
        with open(out_dir / "per_query.jsonl", encoding="utf-8") as f:
            all_rows = [json.loads(line) for line in f if line.strip()]
        done = {(s["chunking"], s["retrieval"]) for s in summaries}
        print(f"Resuming: {len(done)} configuration(s) already in {out_dir}")

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = "unknown"
    env = {
        "name": name, "date": dt.datetime.now().isoformat(timespec="seconds"), "git_sha": sha,
        "python": platform.python_version(), "platform": platform.platform(),
        "embedder": getattr(embedder, "name", args.embedder), "llm_model": settings.LLM_MODEL,
        "cost_model": COST_MODEL,
        "llm_top_k": top_k_for_style(style), "chars_per_token": CHARS_PER_TOKEN,
        "embedding_model_for_cost": settings.EMBEDDING_MODEL, "has_llm_key": settings.has_llm_key,
        "n_questions": len(items), "n_answerable": sum(i.answerable for i in items),
        "bm25_cache": not args.no_bm25_cache, "generate": args.generate, "judge": args.judge if args.generate else None,
        "argv": sys.argv[1:] if argv is None else argv,
        "retrieval_configs": {r: asdict(RETRIEVAL_CONFIGS[r]) for r in runnable},
        "chunking_configs": {c: CHUNKING_CONFIGS[c] for c in chunk_names},
    }

    def save(calibration):
        """Writes all outputs; called after every chunking config so an interrupted run keeps its work."""
        pvals = significance(summaries, all_rows)
        (out_dir / "summary.json").write_text(json.dumps(
            {"environment": env, "skipped": skipped, "runs": summaries, "judge_calibration": calibration,
             "p_values_vs_baseline": pvals}, indent=2), encoding="utf-8")
        with open(out_dir / "per_query.jsonl", "w", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
        write_markdown(out_dir / "summary.md", env, summaries, skipped, calibration, pvals)

    for c_name in chunk_names:
        pending = [r for r in runnable if (c_name, r) not in done]
        if not pending:
            continue
        store, chunks, stats = build_index(corpus, CHUNKING_CONFIGS[c_name], embedder,
                                           cache_bm25=not args.no_bm25_cache)
        max_recall = round(M.mean([M.max_achievable_recall(it, chunks) for it in items if it.answerable]), 4)
        print(f"[{c_name}] {stats['n_chunks']} chunks, max achievable recall {max_recall:.3f}", flush=True)
        for r_name in pending:
            t0 = time.perf_counter()
            rows = run_retrieval(items, store, chunks, RETRIEVAL_CONFIGS[r_name], style, args.generate,
                                 judge, c_name, r_name)
            s = summarize(rows, items, stats, max_recall)
            summaries.append(s)
            all_rows.extend(rows)
            print(f"   {r_name:<28} recall@5={s['metrics']['evidence_recall@5']:.3f} "
                  f"MRR={s['metrics']['mrr']:.3f} p50={s['latency_ms']['retrieval_p50']:.1f}ms "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        save([])

    calibration = [calibrate(get_judge(j), items, corpus) for j in (args.calibrate_judge or [])]
    save(calibration)
    print(f"\nWrote {out_dir}/summary.md, summary.json, per_query.jsonl")
    for k, v in skipped.items():
        print(f"Skipped {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
