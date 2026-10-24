"""Embed every image of a verified subset with one encoder (the notebook's steps 3, 5, 6 and 8, without plots).

Usage: python phase2/embed_subset.py --subset phase2/data/atlas_fullhist --encoder resnet50 \
           --out phase2/results/fullhist/resnet50
Writes embeddings.npz (embeddings, image_key, encoder), embedding_index.csv and run_info.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oct_image as OI  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--encoder", choices=OI.ENCODERS, default="resnet50")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    import torch
    torch.manual_seed(42)
    m = OI.load_manifest(a.subset)
    ok = OI.verify_checksums(m, a.subset)
    if not ok.all():
        raise SystemExit(f"checksum mismatch: {m[~ok].image_key.tolist()}")
    X = np.stack([OI.preprocess(OI.load_gray(a.subset / p)[0])[1] for p in m.local_path])
    device = OI.pick_device()
    model, dim = OI.load_encoder(a.encoder, device)
    t0 = time.time()
    E = OI.embed(model, X, device)
    secs = time.time() - t0
    assert E.shape == (len(m), dim) and not np.isnan(E).any()

    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "embeddings.npz", embeddings=E.astype(np.float32), image_key=m.image_key.to_numpy().astype(str),
             encoder=np.array(a.encoder))
    m.drop(columns=["local_path"]).to_csv(a.out / "embedding_index.csv", index=False)
    info = dict(encoder=a.encoder, subset=a.subset.as_posix(), n=len(m), dim=dim, device=str(device),
                seconds=round(secs, 2), torch=torch.__version__)
    (a.out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
