import json
from typing import Iterator, Optional
import httpx
from app.core.config import settings
from app.utils.logging import logger
from app.utils.retry import call_with_retry, describe_error, error_status_code

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class LLMError(RuntimeError):
    """LLM call failed; the message is safe to show to users."""


class LLMClient:
    provider = "none"
    model = ""

    @property
    def description(self) -> str:
        return f"{self.provider}:{self.model}"

    def generate(self, system: str, prompt: str, json_mode: bool = False, fast: bool = False) -> str:
        raise NotImplementedError

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        raise NotImplementedError


class GeminiLLM(LLMClient):
    provider = "gemini"

    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=settings.LLM_TIMEOUT_SECONDS * 1000),
        )
        self._thinking_control_supported = True

    def _config(self, system: str, json_mode: bool = False, fast: bool = False):
        kwargs = {"system_instruction": system, "temperature": settings.LLM_TEMPERATURE}
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        if fast and self._thinking_control_supported:
            # Small utility calls (query rewriting) do not benefit from "thinking" latency.
            kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)
        return genai_types.GenerateContentConfig(**kwargs)

    def generate(self, system: str, prompt: str, json_mode: bool = False, fast: bool = False) -> str:
        def call(use_fast: bool):
            return call_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model, contents=prompt, config=self._config(system, json_mode, use_fast)),
                what="Gemini generation",
                max_total_wait=settings.LLM_MAX_RETRY_WAIT_SECONDS,
            )

        try:
            try:
                response = call(fast)
            except Exception as e:
                if fast and error_status_code(e) == 400:
                    self._thinking_control_supported = False  # model rejects thinking_budget=0
                    response = call(False)
                else:
                    raise
        except Exception as e:
            raise LLMError(describe_error(e)) from e
        text = (response.text or "").strip()
        if not text:
            raise LLMError("the model returned an empty response (it may have been blocked by safety filters)")
        return text

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        def start():
            iterator = iter(self.client.models.generate_content_stream(
                model=self.model, contents=prompt, config=self._config(system)))
            return iterator, next(iterator, None)

        try:
            iterator, first = call_with_retry(start, what="Gemini streaming",
                                              max_total_wait=settings.LLM_MAX_RETRY_WAIT_SECONDS)
            if first is not None and first.text:
                yield first.text
            for chunk in iterator:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(describe_error(e)) from e


class OpenAICompatibleLLM(LLMClient):
    """Works with OpenAI and any server exposing /v1/chat/completions (Ollama, LM Studio, vLLM...)."""
    provider = "openai"

    def __init__(self, base_url: str, api_key: Optional[str], model: str):
        self.model = model
        self.base_url = base_url
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS, headers=headers)

    def _payload(self, system: str, prompt: str, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": settings.LLM_TEMPERATURE,
            "stream": stream,
        }

    def generate(self, system: str, prompt: str, json_mode: bool = False, fast: bool = False) -> str:
        def call():
            r = self.http.post(f"{self.base_url}/chat/completions", json=self._payload(system, prompt, False))
            r.raise_for_status()
            return r.json()

        try:
            data = call_with_retry(call, what="LLM generation", max_total_wait=settings.LLM_MAX_RETRY_WAIT_SECONDS)
            text = (data["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            raise LLMError(describe_error(e)) from e
        if not text:
            raise LLMError("the model returned an empty response")
        return text

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        try:
            with self.http.stream("POST", f"{self.base_url}/chat/completions",
                                  json=self._payload(system, prompt, True)) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
        except Exception as e:
            raise LLMError(describe_error(e)) from e


def create_llm() -> Optional[LLMClient]:
    provider = settings.LLM_PROVIDER
    try:
        if provider == "gemini":
            if not GENAI_AVAILABLE:
                raise LLMError("google-genai is not installed (pip install google-genai)")
            if not settings.GEMINI_API_KEY:
                raise LLMError("no GEMINI_API_KEY configured")
            return GeminiLLM(settings.GEMINI_API_KEY, settings.LLM_MODEL)
        if provider == "openai":
            return OpenAICompatibleLLM(settings.OPENAI_BASE_URL, settings.OPENAI_API_KEY, settings.LLM_MODEL)
    except Exception as e:
        logger.warning(f"LLM provider '{provider}' unavailable: {e}")
    return None


llm: Optional[LLMClient] = create_llm()
