"""Independent v2 paraphrases: deterministic dev/locked split and hash manifest.

The raw file was written by a separate agent that saw only the plain-English type definitions
(no parser code, keyword lists, or existing templates). The split and hashes are fixed once with
`python -m oct_retrieval.paraphrases --freeze`; later runs verify against the manifest.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from . import config

DATA_DIR = config.ROOT / "tests" / "data"
RAW = DATA_DIR / "paraphrases_v2_raw.jsonl"
DEV = DATA_DIR / "paraphrases_v2_dev.jsonl"
LOCKED = DATA_DIR / "paraphrases_v2_locked.jsonl"
MANIFEST = DATA_DIR / "paraphrases_v2_manifest.json"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split(rows: list[dict], seed: int = config.SEED) -> tuple[list[dict], list[dict]]:
    """Half of each qtype to dev, half to locked (seeded, per type)."""
    rng = np.random.default_rng(seed)
    dev, locked = [], []
    for qtype in sorted({r["qtype"] for r in rows}):
        group = [r for r in rows if r["qtype"] == qtype]
        order = rng.permutation(len(group))
        half = len(group) // 2
        dev += [group[i] for i in sorted(order[:half])]
        locked += [group[i] for i in sorted(order[half:])]
    return dev, locked


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def freeze():
    if MANIFEST.exists():
        raise SystemExit(f"{MANIFEST.name} exists; the split is frozen. Delete it deliberately to re-freeze.")
    dev, locked = split(read_jsonl(RAW))
    _write_jsonl(DEV, dev)
    _write_jsonl(LOCKED, locked)
    manifest = {name: {"file": p.name, "sha256": sha256(p), "n": len(read_jsonl(p))}
                for name, p in (("raw", RAW), ("dev", DEV), ("locked", LOCKED))}
    manifest["seed"] = config.SEED
    manifest["note"] = "locked set: evaluate only; never used to change the parser"
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def verify() -> dict:
    """Raise if any paraphrase file changed since freezing."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, p in (("raw", RAW), ("dev", DEV), ("locked", LOCKED)):
        if sha256(p) != manifest[name]["sha256"]:
            raise ValueError(f"{p.name} changed since it was frozen")
    return manifest


def dev_by_type() -> dict[str, list[str]]:
    """Development paraphrases only. Phase 1.3 development must use this, never the locked set."""
    verify()
    out: dict[str, list[str]] = {}
    for r in read_jsonl(DEV):
        out.setdefault(r["qtype"], []).append(r["text"])
    return out


def locked_by_type() -> dict[str, list[str]]:
    verify()
    out: dict[str, list[str]] = {}
    for r in read_jsonl(LOCKED):
        out.setdefault(r["qtype"], []).append(r["text"])
    return out


if __name__ == "__main__":
    freeze() if "--freeze" in sys.argv else print(verify())
