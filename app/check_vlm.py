"""Check the configured vision-language model before a demo: lists image-capable models (OpenAI-compatible providers
that publish /models, e.g. OpenRouter and Ollama) and sends ONE small request with a real OCT B-scan from the dataset.

  python -m app.check_vlm                 # uses .env / environment
  python -m app.check_vlm --list-only     # only list models (no generation request)
The API key is never printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.vlm import OpenAICompatibleVLM, make_vlm  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    print(cfg)
    vlm = make_vlm(cfg.provider, cfg.model, cfg.base_url, cfg.api_key, **cfg.vlm_options())
    if vlm is None:
        print("OCT_VLM_PROVIDER=evidence: no model to check.")
        return 0
    print(f"{vlm!r}; context budget: {getattr(vlm, 'context_tokens', 0) or 'none'} tokens")
    if isinstance(vlm, OpenAICompatibleVLM):
        try:
            models = vlm.list_models()
            vision = [m for m in models if "image" in (m.get("architecture") or {}).get("input_modalities", [])
                      or "vl" in m["id"].lower()]
            print(f"{len(models)} models listed; {len(vision)} declare image input"
                  + (" (the provider does not publish modalities)" if models and not vision else ""))
            for m in sorted(vision, key=lambda m: m["id"]):
                if m["id"].endswith(":free") or "qwen" in m["id"].lower():
                    print(f"  {m['id']}  context={m.get('context_length')}")
            if models and cfg.model not in {m["id"] for m in models}:
                print(f"WARNING: configured model {cfg.model!r} is not in the provider's list")
        except Exception as ex:   # listing is informational only
            print(f"could not list models: {type(ex).__name__}")
    if a.list_only:
        return 0
    img = next((ROOT / "phase2/data/atlas_fullhist/images").glob("*.webp"))
    reply = vlm.generate("You describe images briefly.", [dict(type="image", path=str(img)), dict(
        type="text", text="In one sentence: what kind of medical image is this?")])
    print(f"finish={reply.finish}" + (f"; usage={reply.usage}" if reply.usage else ""))
    print(reply.error if reply.finish == "error" else reply.text)
    return 0 if reply.finish in ("stop", "length") else 1


if __name__ == "__main__":
    sys.exit(main())
