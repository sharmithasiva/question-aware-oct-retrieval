"""Per-eye figure for full-history subsets: visit x visit cosine matrix of image embeddings next to CST/BCVA over time.

Usage: python phase2/plot_fullhist.py --emb phase2/results/fullhist/resnet50/embeddings.npz \
           --subset phase2/data/atlas_fullhist
Writes fullhist_eyes.png next to the embeddings.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oct_image as OI  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--subset", type=Path, required=True)
    a = ap.parse_args(argv)
    d = np.load(a.emb, allow_pickle=False)
    m = OI.load_manifest(a.subset).set_index("image_key").loc[list(d["image_key"])].reset_index()
    vemb, vis = OI.visit_embeddings(d["embeddings"], m)
    eyes = list(dict.fromkeys(vis.sort_values(["trial", "eye_id"]).eye_id))
    fig, ax = plt.subplots(len(eyes), 2, figsize=(13, 4.2 * len(eyes)), gridspec_kw={"width_ratios": [1, 1.5]})
    for i, e in enumerate(eyes):
        g = vis[vis.eye_id == e].sort_values("visit_index")
        z = vemb[g.index.to_numpy()]
        s = z @ z.T
        im = ax[i, 0].imshow(s, cmap="viridis")
        plt.colorbar(im, ax=ax[i, 0], fraction=.046)
        ax[i, 0].set_title(f"eye {e} ({g.trial.iloc[0]}, {g.split.iloc[0]}): visit x visit cosine", fontsize=9)
        ax[i, 0].set_xlabel("visit position"); ax[i, 0].set_ylabel("visit position")
        x = g.visit_index.to_numpy()
        ax[i, 1].plot(x, g.cst, "o-", color="tab:blue", label="CST (µm)")
        ax[i, 1].set_ylabel("CST (µm)", color="tab:blue")
        t = ax[i, 1].twinx()
        t.plot(x, g.bcva, "s--", color="tab:orange", label="BCVA (letters)")
        t.plot(x, np.r_[np.nan, np.diag(s, -1)] * 100, "^-", color="tab:green", label="cos to previous visit x100")
        t.plot(x, s[0] * 100, "v:", color="tab:gray", label="cos to first visit x100")
        t.set_ylabel("BCVA / cosine x100")
        h1, l1 = ax[i, 1].get_legend_handles_labels(); h2, l2 = t.get_legend_handles_labels()
        ax[i, 1].legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
        ax[i, 1].set_xlabel("visit position (chronological)")
        ax[i, 1].set_title(f"eye {e}: metadata trajectory and image similarity", fontsize=9)
    fig.suptitle(f"Full-history central B-scans, encoder {d['encoder']} (exploratory, not an evaluation)")
    plt.tight_layout()
    out = a.emb.parent / "fullhist_eyes.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
