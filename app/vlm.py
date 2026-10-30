"""Vision-language model adapters. Each turns (system prompt, provider-neutral parts) into one model call.

parts: [{"type": "text", "text": ...} | {"type": "image", "path": ...}] exactly as built by phase2/answer.request_parts,
so a model only ever sees the retrieved evidence package. Adapters never raise for API problems: they return
VLMReply(finish="error", error=<redacted message>) and the answer layer reports it without substituting an answer.

  OpenAICompatibleVLM  POST {base_url}/chat/completions with image_url data URIs (PNG). Works with OpenRouter, Ollama
                       (http://localhost:11434/v1), vLLM, LM Studio and other OpenAI-compatible servers. Standard
                       library only (urllib), so no extra dependency.
  OllamaVLM            Ollama's native POST {base_url}/api/chat, which accepts options.num_ctx (the context window;
                       Ollama's OpenAI-compatible /v1 endpoint ignores it).
  AnthropicVLM         Anthropic Messages API through the `anthropic` SDK (optional dependency).

Context budget (OpenAI-compatible and Ollama adapters): images are converted to greyscale and downscaled so the longer
side is at most image_max_side (never upscaled; sides rounded to the 28-px Qwen2.5-VL patch grid), and
estimate_tokens() gives a conservative prompt size so phase2/answer.answer_vlm can drop optional images or shrink
them until the request fits context_tokens; the answer length is capped to what is left.
"""
from __future__ import annotations

import base64
import functools
import io
import json
import math
import re
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


PATCH = 28                    # Qwen2.5-VL: 14-px patches merged 2x2


@functools.lru_cache(maxsize=256)
def _prepared(path: str, mtime: float, max_side: int | None, fmt: str) -> tuple[bytes, int, int]:
    im = Image.open(path).convert("L")
    w, h = im.size
    if max_side and max(w, h) > max_side:
        s = max_side / max(w, h)
        w, h = max(PATCH, int(w * s) // PATCH * PATCH), max(PATCH, int(h * s) // PATCH * PATCH)
        im = im.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "jpeg":
        im.save(buf, "JPEG", quality=92)
    else:
        im.save(buf, "PNG", optimize=True)
    return buf.getvalue(), w, h


def prepare_image(path: str | Path, max_side: int | None = None, fmt: str = "png") -> tuple[bytes, int, int]:
    """Greyscale image bytes (PNG, lossless, or JPEG quality 92) downscaled so the longer side is at most max_side,
    with its final width and height. The retinal layers keep their aspect ratio; images are never upscaled."""
    path = str(path)
    return _prepared(path, Path(path).stat().st_mtime, max_side, "jpeg" if fmt == "jpeg" else "png")


def png_data_uri(path: str | Path, max_side: int | None = None, fmt: str = "png") -> str:
    """Re-encode any supported image (WebP, PNG, JPEG) as a greyscale data URI: accepted by every provider tested."""
    data, _, _ = prepare_image(path, max_side, fmt)
    return f"data:image/{'jpeg' if fmt == 'jpeg' else 'png'};base64," + base64.b64encode(data).decode()


class BudgetMixin:
    """Context budget shared by the HTTP adapters. context_tokens=0 means no budget is enforced."""
    context_tokens: int = 0
    image_max_side: int | None = 448
    image_min_side: int = 280
    image_tokens: int = 0          # fixed tokens per image (calibration); 0 = estimate from the pixel size
    image_format: str = "png"
    chars_per_token: float = 3.0   # conservative for English text with many numbers and visit IDs
    token_scale: float = 1.0       # raised when the server counts more tokens than estimated (learn_* below)

    def _budget(self, context_tokens=0, image_max_side=448, image_min_side=280, image_tokens=0, image_format="png"):
        self.context_tokens = int(context_tokens or 0)
        self.image_max_side = int(image_max_side) if image_max_side else None
        self.image_min_side = int(image_min_side or 0)
        self.image_tokens, self.image_format = int(image_tokens or 0), image_format

    def image_sides(self) -> list[int | None]:
        """Image sizes to try, largest first (answer._fit_budget shrinks images only after dropping optional ones)."""
        if not self.image_max_side:
            return [None]
        out, side = [], self.image_max_side
        while side >= max(self.image_min_side, PATCH):
            out.append(side)
            side -= 2 * PATCH
        return out or [self.image_max_side]

    def image_cost(self, path: str, side: int | None) -> int:
        if self.image_tokens:
            return self.image_tokens
        _, w, h = prepare_image(path, side, self.image_format)
        # one token per 14x14 patch: deliberately above Qwen2.5-VL's 2x2-merged count, which Ollama exceeds in practice
        return math.ceil(w / 14) * math.ceil(h / 14) + 4

    def estimate_tokens(self, system: str, parts: list[dict], side: int | None = None) -> int:
        side = side if side is not None else self.image_max_side
        chars = len(system) + sum(len(p["text"]) for p in parts if p["type"] == "text")
        raw = (math.ceil(chars / self.chars_per_token) + 32
               + sum(self.image_cost(p["path"], p.get("max_side", side)) for p in parts if p["type"] == "image"))
        return math.ceil(raw * self.token_scale)

    def learn_overflow(self, system: str, parts: list[dict], error: str) -> bool:
        """After the server rejected a request for its size: make the next estimate at least what the server counted
        (or 50 % larger when it gave no count) and lower the budget to the server's limit when it named one. Returns
        False when the error is not about the context window (nothing learned, no retry)."""
        if not CONTEXT_RE.search(error or ""):
            return False
        est = self.estimate_tokens(system, parts)
        sent_window = {self.context_tokens, getattr(self, "num_ctx", 0)}
        nums = [int(n.replace(",", "")) for n in TOKENS_RE.findall(error)]
        if len(nums) >= 2 and max(nums) in sent_window:      # "requested context 8192 ... max 4096": no prompt count
            nums = [min(nums)]
        need = max(nums) if len(nums) >= 2 else (nums[0] if nums and nums[0] > est else None)
        limit = min(nums) if len(nums) >= 2 else (nums[0] if nums and nums[0] <= est else None)
        if limit:
            self.context_tokens = min(self.context_tokens or limit, limit)
            if getattr(self, "num_ctx", 0):
                self.num_ctx = min(self.num_ctx, limit)
        if need and need > est:
            self.token_scale *= need / est * 1.05
        elif not limit or limit >= est:
            self.token_scale *= 1.5
        return True

    def learn_usage(self, system: str, parts: list[dict], prompt_tokens) -> None:
        """Calibrate from the prompt size the server reports for a successful request (only ever upwards)."""
        est = self.estimate_tokens(system, parts)
        if prompt_tokens and int(prompt_tokens) > est:
            self.token_scale *= int(prompt_tokens) / est * 1.05

    def output_tokens(self, system: str, parts: list[dict]) -> int:
        if not self.context_tokens:
            return self.max_tokens
        return max(1, min(self.max_tokens, self.context_tokens - self.estimate_tokens(system, parts)))


HTTP_HINTS = {401: "authentication failed: check the API key", 403: "access denied by the provider",
              402: "payment required: free credits may be exhausted", 404: "model or endpoint not found",
              413: "request too large", 429: "rate limited by the provider: wait and retry, or switch provider",
              400: "request rejected (the model may not accept images)"}
CONTEXT_HINT = ("the request is larger than the model's context window: lower OCT_VLM_MAX_IMAGES or "
                "OCT_VLM_IMAGE_MAX_SIDE, or raise OCT_VLM_NUM_CTX (Ollama) together with OCT_VLM_CONTEXT")


CONTEXT_RE = re.compile(r"context|too (long|many tokens)|exceed|num_ctx|maximum.*tokens", re.I)
TOKENS_RE = re.compile(r"(\d[\d,]*)\s*(?:input\s+)?tokens?\b", re.I)


def _context_hint(detail: str) -> str:
    return f" ({CONTEXT_HINT})" if CONTEXT_RE.search(detail) else ""


class OpenAICompatibleVLM(BudgetMixin):
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0, max_tokens: int = 1500,
                 extra_headers: dict | None = None, opener=None, **budget):
        if not base_url or not model:
            raise ValueError("an OpenAI-compatible provider needs a base URL and a model id")
        self.base_url, self.model = base_url.rstrip("/"), model
        self._key = api_key
        self.timeout, self.max_tokens = timeout, max_tokens
        self.extra_headers = dict(extra_headers or {})
        if "openrouter.ai" in self.base_url:
            self.extra_headers.setdefault("X-Title", "Question-aware OCT retrieval (research demo)")
        self._open = opener or urllib.request.urlopen
        self._budget(**budget)

    def __repr__(self):
        return f"OpenAICompatibleVLM(base_url={self.base_url!r}, model={self.model!r}, key={'set' if self._key else 'none'})"

    def payload(self, system: str, parts: list[dict]) -> dict:
        content = [{"type": "image_url", "image_url": {"url": png_data_uri(
            p["path"], p.get("max_side", self.image_max_side), self.image_format)}} if p["type"] == "image"
                   else {"type": "text", "text": p["text"]} for p in parts]
        return dict(model=self.model, max_tokens=self.output_tokens(system, parts), temperature=0.2,
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
            return A.redact(f"HTTP {ex.code}: {HTTP_HINTS.get(ex.code, 'provider error')}{_context_hint(detail)}. "
                            f"{detail}", 500)
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
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return VLMReply(finish="error", error=A.redact(msg + _context_hint(msg), 500))
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


class OllamaVLM(OpenAICompatibleVLM):
    """Ollama's native chat API, so the context window can be set per request (options.num_ctx). Images go in the
    message's `images` list in order; the text names each one, since they are not interleaved with the text."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0, max_tokens: int = 1500,
                 opener=None, num_ctx: int = 0, **budget):
        base_url = re.sub(r"/v1/?$", "", (base_url or "http://localhost:11434").rstrip("/"))
        super().__init__(base_url, model, api_key, timeout, max_tokens, opener=opener, **budget)
        # the window is always set explicitly, so the server never runs a smaller default than the budget assumes
        self.num_ctx = int(num_ctx or self.context_tokens or 0)

    def __repr__(self):
        return f"OllamaVLM(base_url={self.base_url!r}, model={self.model!r}, num_ctx={self.num_ctx or 'server default'})"

    def payload(self, system: str, parts: list[dict]) -> dict:
        images, lines, label = [], [], None
        for p in parts:
            if p["type"] == "image":
                data, _, _ = prepare_image(p["path"], p.get("max_side", self.image_max_side), self.image_format)
                images.append(base64.b64encode(data).decode())
                lines.append(f"Image {len(images)}: {label or 'B-scan'}")
                label = None
            elif p["text"].startswith("B-scan:"):
                label = p["text"]
            else:
                lines.append(p["text"])
        options = dict(temperature=0.2, num_predict=self.output_tokens(system, parts))
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        user = dict(role="user", content="\n".join(lines))
        if images:
            user["images"] = images
        return dict(model=self.model, stream=False, options=options,
                    messages=[{"role": "system", "content": system}, user])

    def generate(self, system: str, parts: list[dict]) -> VLMReply:
        try:
            out = self._request("/api/chat", self.payload(system, parts))
        except Exception as ex:
            return VLMReply(finish="error", error=self._error(ex))
        if out.get("error"):
            msg = str(out["error"])
            return VLMReply(finish="error", error=A.redact(msg + _context_hint(msg), 500))
        text = (out.get("message") or {}).get("content") or ""
        finish = "length" if out.get("done_reason") == "length" else "stop"
        return VLMReply(text=text, finish=finish, usage=dict(prompt_tokens=out.get("prompt_eval_count"),
                                                             completion_tokens=out.get("eval_count")))

    def list_models(self) -> list[dict]:
        return [dict(id=m.get("name") or m.get("model")) for m in self._request("/api/tags").get("models", [])]


def is_ollama(provider: str, base_url: str) -> bool:
    return provider == "ollama" or ":11434" in (base_url or "")


def default_context(provider: str, base_url: str, context_tokens: int | None, num_ctx: int) -> int:
    """The token budget: OCT_VLM_CONTEXT, else OCT_VLM_NUM_CTX, else 4096 for Ollama, else none (0)."""
    if context_tokens is not None:
        return int(context_tokens)
    return int(num_ctx or (4096 if is_ollama(provider, base_url) else 0))


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
             max_tokens: int = 1500, context_tokens: int | None = None, num_ctx: int = 0,
             image_max_side: int | None = 448, image_min_side: int = 280, image_tokens: int = 0,
             image_format: str = "png"):
    """Adapter for the provider, or None for evidence-only mode. context_tokens=None picks a default: num_ctx when
    set, else 4096 for Ollama, else no budget. Any Ollama URL (also .../v1) gets the native Ollama adapter."""
    if provider == "evidence":
        return None
    context_tokens = default_context(provider, base_url, context_tokens, num_ctx)
    budget = dict(context_tokens=context_tokens, image_max_side=image_max_side, image_min_side=image_min_side,
                  image_tokens=image_tokens, image_format=image_format)
    if provider == "ollama" or (provider == "openai_compatible" and is_ollama(provider, base_url)):
        # Ollama's OpenAI-compatible /v1 cannot set the context window, so its native API is used for every Ollama
        # URL; num_ctx (= the budget unless set) is sent with every request.
        if num_ctx and context_tokens and num_ctx < context_tokens:
            context_tokens = budget["context_tokens"] = num_ctx
        return OllamaVLM(base_url, model, api_key, timeout, max_tokens, num_ctx=num_ctx, **budget)
    if provider == "openai_compatible":
        return OpenAICompatibleVLM(base_url, model, api_key, timeout, max_tokens, **budget)
    if provider == "anthropic":
        return AnthropicVLM(model, api_key, max_tokens)
    raise ValueError(f"unknown provider {provider!r}")
