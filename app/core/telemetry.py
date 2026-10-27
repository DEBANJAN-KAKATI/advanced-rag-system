"""
Per-query tracing of latency, token usage and cost.

Pipeline code calls `stage()`, `record_llm_call()` and `record_embedding()`
unconditionally; they are no-ops unless a trace is active. A trace is opened
with `with trace() as t:` around one query (the chat endpoint does this, and so
does the evaluation harness), and `t.summary()` gives the numbers.
"""
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional

from app.core.pricing import cost_usd, estimate_tokens

_current: ContextVar[Optional["QueryTrace"]] = ContextVar("rag_query_trace", default=None)


class QueryTrace:
    def __init__(self):
        self._start = time.perf_counter()
        self.total_ms: Optional[float] = None
        self.stages: Dict[str, float] = defaultdict(float)
        self.llm_calls: List[Dict[str, Any]] = []
        self.embedding_calls: List[Dict[str, Any]] = []
        self.annotations: Dict[str, Any] = {}

    def finish(self):
        if self.total_ms is None:
            self.total_ms = (time.perf_counter() - self._start) * 1000

    def summary(self) -> Dict[str, Any]:
        llm_in = sum(c["input_tokens"] for c in self.llm_calls)
        llm_out = sum(c["output_tokens"] for c in self.llm_calls)
        emb_tokens = sum(c["input_tokens"] for c in self.embedding_calls)
        costs = [c["cost_usd"] for c in self.llm_calls + self.embedding_calls]
        known = [c for c in costs if c is not None]
        return {
            "total_ms": round(self.total_ms if self.total_ms is not None else
                              (time.perf_counter() - self._start) * 1000, 2),
            "stages_ms": {k: round(v, 2) for k, v in self.stages.items()},
            "llm_calls": len(self.llm_calls),
            "llm_input_tokens": llm_in,
            "llm_output_tokens": llm_out,
            "embedding_calls": len(self.embedding_calls),
            "embedding_tokens": emb_tokens,
            "estimated_cost_usd": round(sum(known), 8) if known or not costs else None,
            "cost_complete": len(known) == len(costs),
            "tokens_estimated": any(c.get("estimated") for c in self.llm_calls + self.embedding_calls),
            "annotations": dict(self.annotations),
        }


def current_trace() -> Optional[QueryTrace]:
    return _current.get()


@contextmanager
def trace() -> Iterator[QueryTrace]:
    """Opens a trace, or reuses the active one so nested callers share it."""
    existing = _current.get()
    if existing is not None:
        yield existing
        return
    t = QueryTrace()
    token = _current.set(t)
    try:
        yield t
    finally:
        t.finish()
        _current.reset(token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Adds the wall time of the block to `name` (accumulates across calls)."""
    t = _current.get()
    start = time.perf_counter()
    try:
        yield
    finally:
        if t is not None:
            t.stages[name] += (time.perf_counter() - start) * 1000


def record_llm_call(purpose: str, model: str, input_tokens: int, output_tokens: int,
                    estimated: bool = False, provider: str = "gemini"):
    t = _current.get()
    if t is None:
        return
    t.llm_calls.append({
        "purpose": purpose,
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "estimated": estimated,
        "cost_usd": cost_usd(model, input_tokens, output_tokens, provider),
    })


def record_llm_response(purpose: str, model: str, response: Any, prompt: str, output_text: str):
    """Records a Gemini generate_content call from its usage_metadata.

    Thinking tokens are billed as output, so they are counted as output here.
    Falls back to a character-based estimate if the response has no usage data.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is not None and getattr(usage, "prompt_token_count", None) is not None:
        output = (getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        record_llm_call(purpose, model, usage.prompt_token_count, output)
    else:
        record_llm_call(purpose, model, estimate_tokens(prompt), estimate_tokens(output_text), estimated=True)


def record_embedding(provider: str, model: str, n_texts: int, input_tokens: int, estimated: bool = True):
    t = _current.get()
    if t is None:
        return
    t.embedding_calls.append({
        "provider": provider,
        "model": model,
        "n_texts": n_texts,
        "input_tokens": int(input_tokens),
        "estimated": estimated,
        "cost_usd": cost_usd(model, input_tokens, 0, provider),
    })


def annotate(key: str, value: Any):
    t = _current.get()
    if t is not None:
        t.annotations[key] = value
