"""Uploaded OCT scans: validation, exact dataset matching and (optional) embedding with the existing pipeline.

An uploaded image is EXTERNAL by default: it gets no patient, eye, visit, date or history. It is linked to a dataset
visit only when it is byte-identical to a subset image, or pixel-identical after the same decoding the pipeline uses
(oct_image.load_gray), e.g. a dataset WebP saved losslessly as PNG. Nothing approximate is ever called a match.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase2"))
import oct_image as OI  # noqa: E402

ALLOWED_FORMATS = {"PNG": "png", "JPEG": "jpeg"}
MIN_SIDE, MAX_SIDE = 64, 4096


class UploadError(ValueError):
    """Shown to the user as is."""


@dataclass
class UploadedScan:
    filename: str
    sha256: str
    pixel_sha256: str
    width: int
    height: int
    format: str
    path: Path                     # greyscale PNG copy given to the model and shown in the UI
    warnings: list[str] = field(default_factory=list)
    matched_visit: str | None = None
    matched_eye: int | None = None
    matched_image_key: str | None = None
    matched_scan_index: int | None = None

    @property
    def external(self) -> bool:
        return self.matched_visit is None


def pixel_hash(gray: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(gray).tobytes() + repr(gray.shape).encode()).hexdigest()


def validate_upload(data: bytes, filename: str, workdir: Path, max_mb: float = 10.0) -> UploadedScan:
    if not data:
        raise UploadError("the file is empty")
    if len(data) > max_mb * 1024 * 1024:
        raise UploadError(f"the file is larger than {max_mb:g} MB")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        im = Image.open(io.BytesIO(data))
        im.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as ex:
        raise UploadError(f"not a readable image ({type(ex).__name__})") from None
    if im.format not in ALLOWED_FORMATS:
        raise UploadError(f"unsupported format {im.format}; upload a PNG or JPEG B-scan")
    w, h = im.size
    if min(w, h) < MIN_SIDE or max(w, h) > MAX_SIDE:
        raise UploadError(f"image is {w}x{h}; sides must be between {MIN_SIDE} and {MAX_SIDE} pixels")
    warnings = []
    rgb = np.asarray(im.convert("RGB")).astype(np.int16)
    if max(np.abs(rgb[..., 0] - rgb[..., 1]).max(), np.abs(rgb[..., 1] - rgb[..., 2]).max()) > 24:
        warnings.append("the image has colour; OCT B-scans are normally greyscale, so it may not be an OCT B-scan "
                        "(it is converted to greyscale)")
    if not 0.5 <= w / h <= 3.0:
        warnings.append(f"unusual aspect ratio {w / h:.2f} for a B-scan")
    workdir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(data).hexdigest()
    raw = workdir / f"upload_{sha[:16]}.{ALLOWED_FORMATS[im.format]}"
    raw.write_bytes(data)
    gray, _ = OI.load_gray(raw)                        # the pipeline's own decoding
    out = workdir / f"upload_{sha[:16]}_gray.png"
    Image.fromarray(gray).save(out, "PNG")
    return UploadedScan(filename=Path(filename).name, sha256=sha, pixel_sha256=pixel_hash(gray), width=w, height=h,
                        format=im.format, path=out, warnings=warnings)


class DatasetImageIndex:
    """sha256 and pixel hash -> manifest row for every image of a subset; cached on disk per manifest version."""

    def __init__(self, subset: Path, cache_dir: Path):
        self.subset = Path(subset)
        man = OI.load_manifest(self.subset)
        mf = hashlib.sha256((self.subset / "manifest.csv").read_bytes()).hexdigest()[:16]
        cache = Path(cache_dir) / f"pixel_index_{self.subset.name}_{mf}.json"
        rows = {r.image_key: r for r in man.itertuples()}
        self.by_sha = {r.sha256: r for r in man.itertuples()}
        if cache.exists():
            pix = json.loads(cache.read_text())
        else:
            pix = {r.image_key: pixel_hash(OI.load_gray(self.subset / r.local_path)[0]) for r in man.itertuples()
                   if (self.subset / r.local_path).exists()}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(pix))
        self.by_pixels = {h: rows[k] for k, h in pix.items() if k in rows}

    def match(self, scan: UploadedScan) -> UploadedScan:
        r = self.by_sha.get(scan.sha256) or self.by_pixels.get(scan.pixel_sha256)
        if r is not None:
            scan.matched_visit, scan.matched_eye = r.visit_id, int(r.eye_id)
            scan.matched_image_key, scan.matched_scan_index = r.image_key, int(r.scan_index)
        return scan


def embed_scan(scan: UploadedScan, encode) -> np.ndarray:
    """L2-normalised embedding with the SAME preprocessing as the stored embeddings (oct_image.preprocess)."""
    gray, _ = OI.load_gray(scan.path)
    x = OI.preprocess(gray)[1][None]
    return OI.l2n(np.asarray(encode(x), dtype=np.float32))[0]
