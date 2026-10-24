"""Phase 2 image pipeline: load verified B-scans, preprocess, embed with a pretrained encoder, rank by cosine.

Used by notebooks/01_image_processing.ipynb and phase2/run_local.py. Does not import or modify Phase 1 code.
torch/torchvision are imported lazily so the loading and ranking helpers work without them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224
ENCODERS = ("resnet50", "dinov2_vits14")


# --- loading -------------------------------------------------------------------------------------------------

def load_manifest(subset_dir: Path) -> pd.DataFrame:
    m = pd.read_csv(Path(subset_dir) / "manifest.csv")
    return m.sort_values(["trial", "eye_id", "visit_index", "scan_index"]).reset_index(drop=True)


def verify_checksums(m: pd.DataFrame, subset_dir: Path) -> pd.Series:
    """True where the file on disk has the SHA-256 recorded at fetch time."""
    return m.apply(lambda r: hashlib.sha256((Path(subset_dir) / r.local_path).read_bytes()).hexdigest() == r.sha256,
                   axis=1)


def load_gray(path: Path) -> tuple[np.ndarray, dict]:
    """Load one B-scan as uint8 grayscale (H, W). The atlas WebPs decode as RGB; lossy WebP chroma coding means
    the channels are near-identical but not exactly equal, so the max channel difference is reported and the
    image is converted with PIL's luminance transform."""
    im = Image.open(path)
    info = {"mode": im.mode, "width": im.width, "height": im.height}
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    info["max_channel_diff"] = int(max(np.abs(a[..., 0] - a[..., 1]).max(), np.abs(a[..., 1] - a[..., 2]).max()))
    g = np.asarray(im.convert("L"))
    return g.astype(np.uint8), info


# --- preprocessing -------------------------------------------------------------------------------------------

def preprocess(gray: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """uint8 (H, W) -> (resized float in [0, 1] (size, size), normalised float32 (3, size, size)).

    1. Resize to size x size with bicubic interpolation. The source is 504 x 496 (aspect 1.016), so the
       distortion from not padding is ~1.6 % and is accepted.
    2. Scale to [0, 1] by /255 (fixed scaling, not per-image min-max, so brightness differences survive).
    3. Replicate to 3 channels and apply ImageNet mean/std, which both encoders were trained with.
    """
    r = np.asarray(Image.fromarray(gray).resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    x = np.repeat(r[None], 3, axis=0)
    x = (x - np.asarray(IMAGENET_MEAN, np.float32)[:, None, None]) / np.asarray(IMAGENET_STD, np.float32)[:, None, None]
    return r, x.astype(np.float32)


# --- encoders ------------------------------------------------------------------------------------------------

def pick_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(name: str, device):
    """Pretrained, non-gated encoders. Returns (model in eval mode, embedding dim).

    resnet50       torchvision IMAGENET1K_V2 weights (download.pytorch.org); global-average-pooled 2048-d.
    dinov2_vits14  Meta DINOv2 ViT-S/14 via torch.hub (github.com/facebookresearch/dinov2, Apache-2.0); 384-d CLS.
    """
    import torch
    if name == "resnet50":
        import torchvision
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        dim = 2048
    elif name == "dinov2_vits14":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
        dim = 384
    else:
        raise ValueError(f"unknown encoder {name!r}; choose from {ENCODERS}")
    return m.eval().to(device), dim


def embed(model, batch: np.ndarray, device, batch_size: int = 32) -> np.ndarray:
    """(N, 3, H, W) float32 -> (N, D) float32, not normalised."""
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(batch), batch_size):
            out.append(model(torch.from_numpy(batch[i:i + batch_size]).to(device)).float().cpu().numpy())
    return np.concatenate(out)


# --- similarity and image-level sanity metrics ---------------------------------------------------------------

def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    z = l2n(emb)
    return z @ z.T


def topk(sim: np.ndarray, q: int, k: int = 5) -> np.ndarray:
    """Indices of the k most similar items to q, excluding q itself (stable tie-break by index)."""
    s = sim[q].copy()
    s[q] = -np.inf
    return np.argsort(-s, kind="stable")[:k]


@dataclass
class NeighbourStats:
    k: int
    same_eye_at_k: float          # mean fraction of the top-k that share the query's eye
    same_eye_chance: float        # expected fraction under random ranking
    same_visit_at_1: float        # fraction of queries whose top-1 is another scan of the same visit
    same_visit_chance_at_1: float
    same_trial_at_k: float
    same_trial_chance: float


def neighbour_stats(sim: np.ndarray, m: pd.DataFrame, k: int = 5) -> NeighbourStats:
    n = len(m)
    eye, visit, trial = m.eye_id.to_numpy(), m.visit_id.to_numpy(), m.trial.to_numpy()
    se, sv, st, ce, cv, ct = [], [], [], [], [], []
    for q in range(n):
        nb = topk(sim, q, k)
        se.append(np.mean(eye[nb] == eye[q]))
        st.append(np.mean(trial[nb] == trial[q]))
        sv.append(float(visit[nb[0]] == visit[q]))
        ce.append((np.sum(eye == eye[q]) - 1) / (n - 1))
        cv.append((np.sum(visit == visit[q]) - 1) / (n - 1))
        ct.append((np.sum(trial == trial[q]) - 1) / (n - 1))
    return NeighbourStats(k, float(np.mean(se)), float(np.mean(ce)), float(np.mean(sv)), float(np.mean(cv)),
                          float(np.mean(st)), float(np.mean(ct)))


def visit_embeddings(emb: np.ndarray, m: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Mean of the L2-normalised scan embeddings of each visit, re-normalised. Returns (V, D) and a visit table."""
    z = l2n(emb)
    vis = (m.groupby("visit_id", sort=False)
           .agg(eye_id=("eye_id", "first"), trial=("trial", "first"), split=("split", "first"),
                visit_index=("visit_index", "first"), week=("week", "first"), cst=("cst", "first"),
                bcva=("bcva", "first"), n_scans=("scan_index", "size"))
           .reset_index())
    v = np.stack([z[(m.visit_id == vid).to_numpy()].mean(0) for vid in vis.visit_id])
    return l2n(v), vis
