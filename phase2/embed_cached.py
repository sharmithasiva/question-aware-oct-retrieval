"""Embed a verified subset in streaming batches, with a persistent cache. Scales to thousands of B-scans.

Same preprocessing and encoders as phase2/oct_image.py (load_gray -> preprocess -> load_encoder), which this script
only imports. Differences from embed_subset.py (kept unchanged for the reported 169-image runs):
  * images are decoded batch by batch, so memory does not grow with the subset (~0.6 MB per image otherwise);
  * embeddings are cached in phase2/cache/<encoder>__<pooling>__prep<hash>/ keyed by the image's SHA-256, so an image
    is never embedded twice (re-runs, overlapping subsets and byte-identical images are free). The key includes a
    hash of the preprocessing code, so changing preprocessing invalidates the cache instead of mixing versions;
  * a decoding audit (mode, size, channel difference) is written to run_info.json;
  * optional pooling variants of the SAME pretrained models (no training):
        resnet50       gap (default, = embed_subset.py) | gem (generalised-mean pooling, p=3, of layer4)
        dinov2_vits14  cls (default, = embed_subset.py) | patchmean (mean of patch tokens) | cls_patchmean (concat)

Outputs (same format as embed_subset.py): <out>/embeddings.npz (embeddings, image_key, encoder), embedding_index.csv,
run_info.json. Default out: phase2/results/<subset name>/<encoder>[__<pooling>]/.

Usage:
  python phase2/embed_cached.py --subset phase2/data/scan_subset_v1 --encoder resnet50
  python phase2/embed_cached.py --subset phase2/data/scan_subset_v1 --encoder dinov2_vits14 --pooling patchmean
  # reproducibility check against the saved 73-image embeddings (expects cosine ~1 for the default pooling):
  python phase2/embed_cached.py --subset phase2/data/atlas_fullhist --encoder resnet50 \
      --out phase2/results/repro_check/fullhist_resnet50 --check-against phase2/results/fullhist/resnet50/embeddings.npz
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
import oct_image as OI  # noqa: E402

POOLINGS = {"resnet50": ("gap", "gem"), "dinov2_vits14": ("cls", "patchmean", "cls_patchmean"),
            "retfound_oct": ("patchmean", "cls")}
DEFAULT_POOLING = {"resnet50": "gap", "dinov2_vits14": "cls", "retfound_oct": "patchmean"}
ENCODER_CHOICES = OI.ENCODERS + ("retfound_oct",)
RETFOUND_ARCH = "vit_large_patch16_224"      # RETFound_mae: ViT-L/16, embed 1024, depth 24, heads 16 (models_vit.py)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_retfound(weights: Path, device, arch: str | None = None, trust_checkpoint: bool = False):
    """RETFound-OCT (Zhou et al., Nature 2023; github.com/rmaphoh/RETFound, CC BY-NC 4.0) as a frozen feature
    extractor. The MAE-pretrained checkpoint (e.g. RETFound_mae_natureOCT.pth from the gated Hugging Face repo
    YukunZhou/RETFound_mae_natureOCT) is loaded into a timm ViT without a classification head; the decoder and mask
    token are dropped. Features are the checkpoint's final-LayerNorm tokens: RETFound's fine-tuning `global_pool`
    averages patch tokens but adds a newly initialised fc_norm, so the pretrained `norm` is used here instead.
    Loading uses torch.load(weights_only=True); a checkpoint that pickles other objects needs trust_checkpoint=True.
    Returns (model, dim)."""
    import timm
    import torch
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=not trust_checkpoint)
    except Exception as ex:
        raise RuntimeError(f"could not load {weights} with weights_only={not trust_checkpoint} ({type(ex).__name__}). "
                           "If it is the official RETFound checkpoint from a source you trust, re-run with "
                           "--trust-checkpoint.") from ex
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()
             if not k.startswith(("decoder", "mask_token", "head", "fc_norm"))}
    arch = arch or RETFOUND_ARCH
    model = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="")
    res = model.load_state_dict(state, strict=False)
    if res.missing_keys:
        raise RuntimeError(f"checkpoint does not match {arch}: {len(res.missing_keys)} missing encoder weights, "
                           f"e.g. {res.missing_keys[:3]}")
    return model.eval().to(device), model.num_features
CACHE_ROOT = ROOT / "phase2" / "cache"


def prep_version() -> str:
    """Hash of the decoding + preprocessing code and constants; part of the cache key."""
    src = inspect.getsource(OI.load_gray) + inspect.getsource(OI.preprocess) + repr(
        (OI.IMAGENET_MEAN, OI.IMAGENET_STD, OI.INPUT_SIZE))
    return hashlib.sha256(src.encode()).hexdigest()[:10]


def make_encoder(name: str, pooling: str, device, weights: Path | None = None, trust_checkpoint: bool = False,
                 arch: str | None = None):
    """Returns (fn: float32 array (N, 3, H, W) -> (N, D) float32, D). Pretrained weights only; eval mode."""
    import torch
    if pooling not in POOLINGS[name]:
        raise ValueError(f"pooling {pooling!r} not available for {name}; choose from {POOLINGS[name]}")
    if name == "retfound_oct":
        if weights is None:
            raise ValueError("retfound_oct needs --weights (the RETFound_mae_natureOCT checkpoint; gated download)")
        model, dim = load_retfound(Path(weights), device, arch, trust_checkpoint)

        def f(x):
            tok = model.forward_features(x)                    # (N, 1 + patches, D), after the pretrained LayerNorm
            return tok[:, 0] if pooling == "cls" else tok[:, model.num_prefix_tokens:].mean(1)
    else:
        model, dim = OI.load_encoder(name, device)
        if name == "resnet50" and pooling == "gem":
            body = torch.nn.Sequential(*list(model.children())[:-2]).eval()   # up to layer4; drops avgpool + fc

            def f(x):
                fm = body(x).clamp(min=1e-6)
                return fm.pow(3).mean((2, 3)).pow(1 / 3)
        elif name == "dinov2_vits14" and pooling != "cls":
            def f(x):
                o = model.forward_features(x)
                pm = o["x_norm_patchtokens"].mean(1)
                return pm if pooling == "patchmean" else torch.cat([o["x_norm_clstoken"], pm], 1)
            dim = dim if pooling == "patchmean" else 2 * dim
        else:
            f = model

    def run(batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return f(torch.from_numpy(batch).to(device)).float().cpu().numpy()
    return run, dim


class Cache:
    """Append-only shards of (sha256, embedding). Loading reads every shard; writing never rewrites one."""

    def __init__(self, path: Path):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self.store: dict[str, np.ndarray] = {}
        for p in sorted(path.glob("shard_*.npz")):
            d = np.load(p, allow_pickle=False)
            self.store.update(zip(d["sha256"].tolist(), d["embeddings"]))

    def add(self, shas: list[str], emb: np.ndarray) -> None:
        n = len(list(self.path.glob("shard_*.npz")))
        np.savez(self.path / f"shard_{n:06d}.npz", sha256=np.array(shas), embeddings=emb.astype(np.float32))
        self.store.update(zip(shas, emb.astype(np.float32)))


def embed_subset(m: pd.DataFrame, subset: Path, encode, cache: Cache, batch_size: int = 64,
                 flush_every: int = 8, log=print) -> tuple[np.ndarray, dict]:
    """Embeddings in manifest order. Only images whose SHA-256 is not cached are decoded and encoded."""
    todo = list(dict.fromkeys(s for s in m.sha256 if s not in cache.store))      # unique, order kept
    path_of = dict(zip(m.sha256, m.local_path))
    audit = dict(n_images=len(m), n_unique_sha256=int(m.sha256.nunique()), n_cached_before=len(m.sha256.unique()) -
                 len(todo), n_embedded_now=len(todo), modes={}, sizes={}, max_channel_diff=0)
    pend_s, pend_e, t0 = [], [], time.time()
    for i in range(0, len(todo), batch_size):
        shas = todo[i:i + batch_size]
        xs = []
        for s in shas:
            g, info = OI.load_gray(subset / path_of[s])
            audit["modes"][info["mode"]] = audit["modes"].get(info["mode"], 0) + 1
            wh = f"{info['width']}x{info['height']}"
            audit["sizes"][wh] = audit["sizes"].get(wh, 0) + 1
            audit["max_channel_diff"] = max(audit["max_channel_diff"], info["max_channel_diff"])
            xs.append(OI.preprocess(g)[1])
        e = encode(np.stack(xs))
        if e.shape[0] != len(shas) or not np.isfinite(e).all():
            raise RuntimeError(f"encoder returned bad output for batch starting at {i}")
        pend_s += shas
        pend_e.append(e)
        if len(pend_e) >= flush_every or i + batch_size >= len(todo):
            cache.add(pend_s, np.concatenate(pend_e))
            pend_s, pend_e = [], []
            log(f"embedded {min(i + batch_size, len(todo))}/{len(todo)} new images ({time.time() - t0:.0f}s)")
    audit["seconds_embedding"] = round(time.time() - t0, 2)
    return np.stack([cache.store[s] for s in m.sha256]), audit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--encoder", choices=ENCODER_CHOICES, default="resnet50")
    ap.add_argument("--weights", type=Path, help="local checkpoint (required for retfound_oct)")
    ap.add_argument("--trust-checkpoint", action="store_true",
                    help="allow torch.load(weights_only=False) for a checkpoint from a trusted source")
    ap.add_argument("--pooling")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    ap.add_argument("--check-against", type=Path, help="saved embeddings.npz to compare (cosine per common image)")
    a = ap.parse_args(argv)
    pooling = a.pooling or DEFAULT_POOLING[a.encoder]
    tag = a.encoder if pooling == DEFAULT_POOLING[a.encoder] else f"{a.encoder}__{pooling}"
    out = a.out or ROOT / "phase2/results" / a.subset.name / tag

    import torch
    torch.manual_seed(42)
    m = OI.load_manifest(a.subset)
    ok = OI.verify_checksums(m, a.subset)
    if not ok.all():
        raise SystemExit(f"checksum mismatch ({(~ok).sum()} images): {m[~ok].image_key.tolist()[:5]}")
    device = OI.pick_device()
    encode, dim = make_encoder(a.encoder, pooling, device, a.weights, a.trust_checkpoint)
    wtag = f"__w{file_sha(a.weights)[:10]}" if a.weights else ""         # separate cache per checkpoint
    cache = Cache(a.cache_root / f"{a.encoder}__{pooling}__prep{prep_version()}{wtag}")
    E, audit = embed_subset(m, a.subset, encode, cache, a.batch_size, log=lambda s: print(s, file=sys.stderr))
    assert E.shape == (len(m), dim)

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "embeddings.npz", embeddings=E.astype(np.float32), image_key=m.image_key.to_numpy().astype(str),
             encoder=np.array(tag))
    m.drop(columns=["local_path"]).to_csv(out / "embedding_index.csv", index=False)
    info = dict(encoder=a.encoder, pooling=pooling, subset=a.subset.as_posix(), n=len(m), dim=dim, device=str(device),
                weights=None if not a.weights else dict(file=a.weights.name, sha256=file_sha(a.weights)),
                batch_size=a.batch_size, prep_version=prep_version(), cache=str(cache.path.relative_to(ROOT))
                if cache.path.is_relative_to(ROOT) else str(cache.path), torch=torch.__version__, audit=audit)
    if a.check_against:
        d = np.load(a.check_against, allow_pickle=False)
        ref = dict(zip(d["image_key"].tolist(), OI.l2n(d["embeddings"])))
        z = OI.l2n(E)
        cos = [float(z[i] @ ref[k]) for i, k in enumerate(m.image_key) if k in ref]
        info["check_against"] = dict(path=str(a.check_against), n_common=len(cos),
                                     min_cosine=round(min(cos), 6) if cos else None,
                                     mean_cosine=round(float(np.mean(cos)), 6) if cos else None)
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
