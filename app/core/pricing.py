"""
Token pricing used to turn per-query token counts into dollar cost.

Prices are USD per 1M tokens for paid-tier, text-only usage. Provider prices
change, so treat this table as configuration: check it against the provider's
pricing page before quoting numbers, and add rows for any model you switch to.
Local models (sentence-transformers / ONNX) cost nothing per call.
"""
from typing import Dict, Optional

PRICES_PER_MTOK: Dict[str, Dict[str, float]] = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-embedding-001": {"input": 0.15, "output": 0.0},
}

LOCAL_PROVIDERS = {"local", "local-onnx", "sentence-transformers"}


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token for English text).

    Used only when the provider does not report usage (e.g. embedding calls)
    or when projecting cost offline; real usage metadata always wins.
    """
    return max(1, (len(text) + 3) // 4) if text else 0


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0, provider: str = "") -> Optional[float]:
    """Dollar cost of a call, 0.0 for local providers, None if the model is unpriced."""
    if provider in LOCAL_PROVIDERS:
        return 0.0
    price = PRICES_PER_MTOK.get(model.replace("models/", ""))
    if price is None:
        return None
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
