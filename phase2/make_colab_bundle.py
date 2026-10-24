"""Build phase2/dist/phase2_colab_bundle.zip: everything notebooks/01_image_processing.ipynb needs in Colab.

Contents: the phase2 code, the verified atlas subsets (4-visit and full-history; images + manifests + provenance), episodes.csv,
visit_images.csv, a read-only copy of the frozen Phase 1 package (src/oct_retrieval) and the notebook.
Only reads repository files; writes only the zip.

Usage: python phase2/make_colab_bundle.py
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    *sorted((ROOT / "phase2").glob("*.py")),
    *sorted(p for p in (ROOT / "phase2" / "data").rglob("*") if p.is_file()),
    ROOT / "data" / "processed" / "episodes.csv",
    ROOT / "data" / "processed" / "visit_images.csv",
    *sorted((ROOT / "src" / "oct_retrieval").glob("*.py")),
    ROOT / "notebooks" / "01_image_processing.ipynb",
]


def main() -> int:
    out = ROOT / "phase2" / "dist" / "phase2_colab_bundle.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in INCLUDE:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB, {len(INCLUDE)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
