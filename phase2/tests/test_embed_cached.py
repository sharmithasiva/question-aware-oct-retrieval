"""Tests for phase2/embed_cached.py: streaming batches, SHA-256 cache, decoding audit (fake encoder; no torch)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase2"))
import embed_cached as EC  # noqa: E402
import oct_image as OI  # noqa: E402

FULLHIST = ROOT / "phase2/data/atlas_fullhist"


class FakeEncoder:
    """Deterministic stand-in: 8x8 average pooling of channel 0 -> 784-d (enough to tell images apart)."""

    def __init__(self):
        self.seen = 0

    def __call__(self, x):
        self.seen += len(x)
        return x[:, 0].reshape(len(x), 28, 8, 28, 8).mean((2, 4)).reshape(len(x), -1).astype(np.float32)


@pytest.fixture
def manifest():
    return OI.load_manifest(FULLHIST).head(20).reset_index(drop=True)


def test_matches_unbatched_pipeline_and_preserves_order(manifest, tmp_path):
    enc = FakeEncoder()
    E, audit = EC.embed_subset(manifest, FULLHIST, enc, EC.Cache(tmp_path), batch_size=7, flush_every=2,
                               log=lambda s: None)
    X = np.stack([OI.preprocess(OI.load_gray(FULLHIST / p)[0])[1] for p in manifest.local_path])
    assert np.allclose(E, FakeEncoder()(X), atol=1e-6)
    n = manifest.sha256.nunique()                          # < 20: the known identical 01-013 W8 / W12 pair
    assert n == 19 and audit["n_embedded_now"] == n and audit["sizes"] == {"504x496": n}


def test_cache_is_reused_across_runs_and_duplicates_embedded_once(manifest, tmp_path):
    m = manifest.copy()
    m.loc[1, "sha256"] = m.loc[0, "sha256"]                 # simulate a byte-identical image
    m.loc[1, "local_path"] = m.loc[0, "local_path"]
    enc = FakeEncoder()
    E1, a1 = EC.embed_subset(m, FULLHIST, enc, EC.Cache(tmp_path), batch_size=8, log=lambda s: None)
    n = m.sha256.nunique()
    assert enc.seen == n == 18 and np.array_equal(E1[0], E1[1])
    enc2 = FakeEncoder()
    E2, a2 = EC.embed_subset(m, FULLHIST, enc2, EC.Cache(tmp_path), batch_size=8, log=lambda s: None)
    assert enc2.seen == 0 and a2["n_cached_before"] == n and np.array_equal(E1, E2)


def test_bad_encoder_output_is_rejected(manifest, tmp_path):
    with pytest.raises(RuntimeError):
        EC.embed_subset(manifest, FULLHIST, lambda x: np.full((len(x), 4), np.nan, np.float32), EC.Cache(tmp_path),
                        log=lambda s: None)


def test_prep_version_is_stable_and_poolings_declared():
    assert EC.prep_version() == EC.prep_version() and len(EC.prep_version()) == 10
    assert all(EC.DEFAULT_POOLING[e] in EC.POOLINGS[e] for e in OI.ENCODERS)


def test_gem_pooling_shapes_with_random_weights(monkeypatch):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")

    def random_resnet(name, device):                        # pretrained weights are not downloaded in tests
        m = torchvision.models.resnet50(weights=None)
        m.fc = torch.nn.Identity()
        return m.eval(), 2048
    monkeypatch.setattr(OI, "load_encoder", random_resnet)
    x = np.random.default_rng(0).normal(size=(2, 3, 224, 224)).astype(np.float32)
    gap, d1 = EC.make_encoder("resnet50", "gap", torch.device("cpu"))
    gem, d2 = EC.make_encoder("resnet50", "gem", torch.device("cpu"))
    a, b = gap(x), gem(x)
    assert a.shape == (2, d1) and b.shape == (2, d2) and np.isfinite(b).all()
    with pytest.raises(ValueError):
        EC.make_encoder("resnet50", "cls", torch.device("cpu"))


def test_cli_end_to_end_with_random_weights(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")

    def random_resnet(name, device):
        torch.manual_seed(0)
        m = torchvision.models.resnet50(weights=None)
        m.fc = torch.nn.Identity()
        return m.eval(), 2048
    monkeypatch.setattr(OI, "load_encoder", random_resnet)
    out, ref = tmp_path / "out", ROOT / "phase2/results/fullhist/resnet50/embeddings.npz"
    args = ["--subset", str(FULLHIST), "--encoder", "resnet50", "--out", str(out), "--cache-root", str(tmp_path / "c"),
            "--batch-size", "16", "--check-against", str(ref)]
    assert EC.main(args) == 0
    d = np.load(out / "embeddings.npz")
    assert d["embeddings"].shape == (73, 2048) and list(d["image_key"]) == list(OI.load_manifest(FULLHIST).image_key)
    import json
    info = json.loads((out / "run_info.json").read_text())
    assert info["check_against"]["n_common"] == 73 and info["audit"]["n_embedded_now"] == 72   # one duplicate pair
    assert EC.main(args) == 0                                         # second run: everything from the cache
    assert json.loads((out / "run_info.json").read_text())["audit"]["n_embedded_now"] == 0
