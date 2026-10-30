"""RETFound-OCT encoder option: checkpoint loading and pooling, with a tiny random ViT in MAE checkpoint format
(the real gated ViT-L weights are not needed; no network)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2"))
import embed_cached as EC  # noqa: E402
import oct_image as OI  # noqa: E402

ARCH = "vit_tiny_patch16_224"


def mae_checkpoint(path: Path, extra=None) -> torch.nn.Module:
    torch.manual_seed(0)
    enc = timm.create_model(ARCH, pretrained=False, num_classes=0, global_pool="").eval()
    state = dict(enc.state_dict())
    state["mask_token"] = torch.zeros(1, 1, enc.num_features)                # MAE-only keys are dropped on load
    state["decoder_embed.weight"] = torch.zeros(8, enc.num_features)
    ckpt = {"model": state, **(extra or {})}
    torch.save(ckpt, path)
    return enc


def test_loads_mae_checkpoint_and_pools(tmp_path):
    enc = mae_checkpoint(tmp_path / "ckpt.pth")
    x = np.random.default_rng(0).normal(size=(2, 3, 224, 224)).astype(np.float32)
    with torch.inference_mode():
        tok = enc.forward_features(torch.from_numpy(x))
    for pooling, want in (("patchmean", tok[:, enc.num_prefix_tokens:].mean(1)), ("cls", tok[:, 0])):
        f, dim = EC.make_encoder("retfound_oct", pooling, torch.device("cpu"), tmp_path / "ckpt.pth", arch=ARCH)
        got = f(x)
        assert got.shape == (2, dim) == (2, enc.num_features) and np.allclose(got, want.numpy(), atol=1e-5)


def test_missing_weights_and_mismatched_checkpoint_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        EC.make_encoder("retfound_oct", "patchmean", torch.device("cpu"))
    torch.save({"model": {"blocks.0.norm1.weight": torch.ones(192)}}, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError, match="missing encoder weights"):
        EC.load_retfound(tmp_path / "bad.pth", torch.device("cpu"), ARCH)


def test_pickled_objects_need_explicit_trust(tmp_path):
    mae_checkpoint(tmp_path / "ckpt.pth", extra={"args": argparse.Namespace(model="RETFound_mae")})
    with pytest.raises(RuntimeError, match="trust-checkpoint"):
        EC.load_retfound(tmp_path / "ckpt.pth", torch.device("cpu"), ARCH)
    model, dim = EC.load_retfound(tmp_path / "ckpt.pth", torch.device("cpu"), ARCH, trust_checkpoint=True)
    assert dim == 192


def test_cli_separate_cache_per_checkpoint(tmp_path, monkeypatch):
    mae_checkpoint(tmp_path / "ckpt.pth")
    monkeypatch.setattr(EC, "RETFOUND_ARCH", ARCH)
    sub = ROOT / "phase2/data/atlas_fullhist"
    out = tmp_path / "out"
    args = ["--subset", str(sub), "--encoder", "retfound_oct", "--weights", str(tmp_path / "ckpt.pth"),
            "--out", str(out), "--cache-root", str(tmp_path / "cache"), "--batch-size", "16"]
    assert EC.main(args) == 0
    info = json.loads((out / "run_info.json").read_text())
    assert info["encoder"] == "retfound_oct" and info["pooling"] == "patchmean" and info["dim"] == 192
    assert info["weights"]["sha256"] == EC.file_sha(tmp_path / "ckpt.pth")
    caches = [p.name for p in (tmp_path / "cache").iterdir()]
    assert len(caches) == 1 and caches[0].startswith("retfound_oct__patchmean__prep") and "__w" in caches[0]
    d = np.load(out / "embeddings.npz")
    assert d["embeddings"].shape == (73, 192) and list(d["image_key"]) == list(OI.load_manifest(sub).image_key)
