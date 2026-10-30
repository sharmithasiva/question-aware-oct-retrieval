"""Configuration from environment variables, optionally read from a `.env` file at the repository root.

A real environment variable always wins over `.env`. API keys are held in memory only and never printed (the
`Config.__repr__` hides them). See `.env.example` for every variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# UI presets: label -> (provider, base_url, default model). All are editable in the sidebar.
PRESETS = {
    "Evidence only (no model)": ("evidence", "", ""),
    "OpenRouter (hosted, OpenAI-compatible)": ("openai_compatible", "https://openrouter.ai/api/v1",
                                               "qwen/qwen2.5-vl-72b-instruct:free"),
    "Ollama (local, OpenAI-compatible)": ("openai_compatible", "http://localhost:11434/v1", "qwen2.5vl:7b"),
    "Custom OpenAI-compatible endpoint": ("openai_compatible", "", ""),
    "Anthropic Claude": ("anthropic", "", "claude-opus-5-5"),
}


@dataclass
class Config:
    provider: str = "evidence"                   # evidence | openai_compatible | anthropic
    model: str = ""
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    timeout: float = 120.0
    max_tokens: int = 1500
    subset: Path = ROOT / "phase2/data/atlas_fullhist"
    embeddings: Path = ROOT / "phase2/results/fullhist/resnet50/embeddings.npz"
    visit_mode: str = "central"
    splits: tuple[str, ...] = ("train",)         # eyes offered in the UI (held-out val/test eyes stay out by default)
    upload_embedding: bool = True                # embed uploads with ResNet-50 for "most similar visit" questions
    upload_max_mb: float = 10.0
    workdir: Path = ROOT / "app/.runtime"        # uploaded-image copies and caches (git-ignored)

    def __repr__(self) -> str:
        shown = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "api_key"}
        return f"Config({shown}, api_key={'set' if self.api_key else 'not set'})"


def read_dotenv(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_config(env: dict | None = None, dotenv: Path | None = ROOT / ".env") -> Config:
    e = {**(read_dotenv(dotenv) if dotenv else {}), **(os.environ if env is None else env)}
    provider = e.get("OCT_VLM_PROVIDER", "evidence")
    key = e.get("OCT_VLM_API_KEY") or (e.get("ANTHROPIC_API_KEY", "") if provider == "anthropic"
                                       else e.get("OPENROUTER_API_KEY", ""))
    c = Config(provider=provider, model=e.get("OCT_VLM_MODEL", ""), base_url=e.get("OCT_VLM_BASE_URL", ""),
               api_key=key, timeout=float(e.get("OCT_VLM_TIMEOUT", 120)), max_tokens=int(e.get("OCT_VLM_MAX_TOKENS", 1500)),
               visit_mode=e.get("OCT_VISIT_MODE", "central"),
               splits=tuple(s.strip() for s in e.get("OCT_SPLITS", "train").split(",") if s.strip()),
               upload_embedding=e.get("OCT_UPLOAD_EMBEDDING", "1") not in ("0", "false", "False"),
               upload_max_mb=float(e.get("OCT_UPLOAD_MAX_MB", 10)))
    if e.get("OCT_SUBSET"):
        c.subset = (ROOT / e["OCT_SUBSET"]) if not Path(e["OCT_SUBSET"]).is_absolute() else Path(e["OCT_SUBSET"])
    if e.get("OCT_EMBEDDINGS"):
        c.embeddings = (ROOT / e["OCT_EMBEDDINGS"]) if not Path(e["OCT_EMBEDDINGS"]).is_absolute() \
            else Path(e["OCT_EMBEDDINGS"])
    return c
