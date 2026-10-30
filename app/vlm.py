"""Vision-language model adapters. Each turns (system prompt, provider-neutral parts) into one model call.

parts: [{"type": "text", "text": ...} | {"type": "image", "path": ...}] exactly as built by phase2/answer.request_parts,
so a model only ever sees the retrieved evidence package. Adapters never raise for API problems: they return
VLMReply(finish="error", error=<redacted message>) and the answer layer reports it without substituting an answer.

  OpenAICompatibleVLM  POST {base_url}/chat/completions with image_url data URIs (PNG). Works with OpenRouter, Ollama
                       (http://localhost:11434/v1), vLLM, LM Studio and other OpenAI-compatible servers. Standard
                       library only (urllib), so no extra dependency.
  AnthropicVLM         Anthropic Messages API through the `anthropic` SDK (optional dependency).
"""
from __future__ import annotations

import base64
import io
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
import answer as A  # noqa: E402


@dataclass
class VLMReply:
    text: str = ""
    finish: str = "stop"          # stop | length | refusal | error
    error: str | None = None
    usage: dict = field(default_factory=dict)


def png_data_uri(path: str | Path) -> str:
    """Re-encode any supported image (WebP, PNG, JPEG) as greyscale PNG: accepted by every provider tested here."""
    buf = io.BytesIO()
    Image.open(path).convert("L").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


HTTP_HINTS = {401: "authentication failed: check the API key", 403: "access denied by the provider",
              402: "payment required: free credits may be exhausted", 404: "model or endpoint not found",
              413: "request too large", 429: "rate limited by the provider: wait and retry, or switch provider",
              400: "request rejected (the model may not accept images)"}


class OpenAICompatibleVLM:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0, max_tokens: int = 1500,
                 extra_headers: dict | None = None, opener=None):
        if not base_url or not model:
            raise ValueError("an OpenAI-compatible provider needs a base URL and a model id")
        self.base_url, self.model = base_url.rstrip("/"), model
        self._key = api_key
        self.timeout, self.max_tokens = timeout, max_tokens
        self.extra_headers = dict(extra_headers or {})
        if "openrouter.ai" in self.base_url:
            self.extra_headers.setdefault("X-Title", "Question-aware OCT retrieval (research demo)")
        self._open = opener or urllib.request.urlopen

    def __repr__(self):
        return f"OpenAICompatibleVLM(base_url={self.base_url!r}, model={self.model!r}, key={'set' if self._key else 'none'})"

    def payload(self, system: str, parts: list[dict]) -> dict:
        content = [{"type": "image_url", "image_url": {"url": png_data_uri(p["path"])}} if p["type"] == "image"
                   else {"type": "text", "text": p["text"]} for p in parts]
        return dict(model=self.model, max_tokens=self.max_tokens, temperature=0.2,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": content}])

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", **self.extra_headers}
        if self._key:
            h["Authorization"] = f"Bearer {self._key}"
        return h

    def _request(self, path: str, body: dict | None = None):
        req = urllib.request.Request(self.base_url + path, data=None if body is None else json.dumps(body).encode(),
                                     headers=self._headers(), method="GET" if body is None else "POST")
        with self._open(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _error(self, ex: Exception) -> str:
        if isinstance(ex, urllib.error.HTTPError):
            try:
                detail = ex.read().decode("utf-8", "replace")[:300]
            except Exception:
                detail = ""
            return A.redact(f"HTTP {ex.code}: {HTTP_HINTS.get(ex.code, 'provider error')}. {detail}")
        if isinstance(ex, urllib.error.URLError):
            return A.redact(f"cannot reach {self.base_url} ({ex.reason})")
        if isinstance(ex, TimeoutError):
            return f"no response from {self.base_url} within {self.timeout:.0f} s"
        return A.redact(f"{type(ex).__name__}: {ex}")

    def generate(self, system: str, parts: list[dict]) -> VLMReply:
        try:
            out = self._request("/chat/completions", self.payload(system, parts))
        except Exception as ex:
            return VLMReply(finish="error", error=self._error(ex))
        if out.get("error"):
            err = out["error"]
            return VLMReply(finish="error", error=A.redact(err.get("message", str(err)) if isinstance(err, dict) else err))
        try:
            choice = out["choices"][0]
            msg = choice["message"].get("content") or ""
            text = msg if isinstance(msg, str) else "".join(p.get("text", "") for p in msg if isinstance(p, dict))
        except (KeyError, IndexError, TypeError):
            return VLMReply(finish="error", error="unexpected response format from the provider")
        reason = choice.get("finish_reason") or "stop"
        finish = {"length": "length", "content_filter": "refusal"}.get(reason, "stop")
        return VLMReply(text=text, finish=finish, usage=out.get("usage") or {})

    def list_models(self) -> list[dict]:
        return self._request("/models").get("data", [])


class AnthropicVLM:
    def __init__(self, model: str = A.MODEL, api_key: str = "", max_tokens: int = 1500, client=None):
        self.model, self.max_tokens = model or A.MODEL, max_tokens
        self._key, self._client = api_key, client

    def __repr__(self):
        return f"AnthropicVLM(model={self.model!r}, key={'set' if self._key else 'env'})"

    def generate(self, system: str, parts: list[dict]) -> VLMReply:
        content = [A._image_block(p["path"]) if p["type"] == "image" else {"type": "text", "text": p["text"]}
                   for p in parts]
        try:
            if self._client is None:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._key) if self._key else anthropic.Anthropic()
            resp = self._client.beta.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system, output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                messages=[{"role": "user", "content": content}])
        except ImportError:
            return VLMReply(finish="error", error="the 'anthropic' package is not installed (pip install anthropic)")
        except Exception as ex:
            return VLMReply(finish="error", error=A.redact(f"{type(ex).__name__}: {ex}"))
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        finish = {"refusal": "refusal", "max_tokens": "length"}.get(resp.stop_reason, "stop")
        return VLMReply(text=text, finish=finish)


def make_vlm(provider: str, model: str = "", base_url: str = "", api_key: str = "", timeout: float = 120.0,
             max_tokens: int = 1500):
    """Adapter for the provider, or None for evidence-only mode."""
    if provider == "evidence":
        return None
    if provider == "openai_compatible":
        return OpenAICompatibleVLM(base_url, model, api_key, timeout, max_tokens)
    if provider == "anthropic":
        return AnthropicVLM(model, api_key, max_tokens)
    raise ValueError(f"unknown provider {provider!r}")
