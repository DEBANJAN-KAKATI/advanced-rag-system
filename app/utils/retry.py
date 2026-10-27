import re
import time
from typing import Callable, TypeVar
from app.utils.logging import logger

T = TypeVar("T")

_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


def error_status_code(exc: BaseException):
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if code is None and response is not None:
        code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def is_retryable(exc: BaseException) -> bool:
    if error_status_code(exc) in _RETRYABLE_CODES:
        return True
    text = str(exc).upper()
    return any(marker in text for marker in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "TIMED OUT", "TIMEOUT"))


def _suggested_delay(exc: BaseException, attempt: int) -> float:
    match = re.search(r"retry(?:Delay|[- ]after)\D{0,10}(\d+(?:\.\d+)?)", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 0.5
    return min(2.0 ** attempt, 16.0)


def call_with_retry(fn: Callable[[], T], what: str, max_total_wait: float, max_attempts: int = 4) -> T:
    """
    Calls fn(), retrying transient failures (rate limits, 5xx, timeouts) with backoff.
    Never waits more than `max_total_wait` seconds in total, so a request cannot hang for minutes.
    """
    waited = 0.0
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_attempts - 1 or not is_retryable(exc):
                raise
            delay = _suggested_delay(exc, attempt)
            if waited + delay > max_total_wait:
                raise
            logger.info(f"{what} failed transiently ({describe_error(exc)}); retrying in {delay:.1f}s")
            time.sleep(delay)
            waited += delay
    raise RuntimeError("unreachable")


def describe_error(exc: BaseException) -> str:
    """Short, user-presentable explanation of an API failure."""
    code = error_status_code(exc)
    text = str(exc)
    upper = text.upper()
    if code == 429 or "RESOURCE_EXHAUSTED" in upper or "QUOTA" in upper:
        return "API quota or rate limit exceeded"
    if code in (401, 403) or "API_KEY_INVALID" in upper or "API KEY NOT VALID" in upper or "PERMISSION_DENIED" in upper:
        return "API key was rejected (check that it is valid)"
    if code == 404 or "NOT_FOUND" in upper:
        return "model not found (check the configured model name)"
    if code in (500, 502, 503, 504) or "UNAVAILABLE" in upper:
        return "the AI service is temporarily unavailable"
    if "TIMEOUT" in upper or "TIMED OUT" in upper:
        return "the request to the AI service timed out"
    if "CONNECT" in upper or "NAME RESOLUTION" in upper or "NETWORK" in upper:
        return "could not connect to the AI service"
    short = text.strip().splitlines()[0][:160] if text.strip() else type(exc).__name__
    return f"{type(exc).__name__}: {short}"
