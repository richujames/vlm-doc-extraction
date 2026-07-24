"""
Build the degraded test split.

The premise of this project: document extraction benchmarks are clean scans,
deployed systems see phone photos. A model that scores well on the first and
collapses on the second is not a working system, and reporting only the first
number hides that.

So we take the SAME test images, apply deterministic corruptions, and score
both. The gap between them is the result that actually matters.

DETERMINISM: seeded per-image by filename, so re-running produces byte-identical
output and clean/degraded are directly comparable. Do not change the seed
between runs and then compare numbers across them.

This file is complete and runnable. Read `DEFAULT_SEVERITY` and decide whether
you agree with the numbers — you will have to defend them.

    python src/degrade.py --input data/test/img --output data/test_degraded/img
    python src/degrade.py --input data/test/img --output data/test_demo/img --limit 5 --save-comparison
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------
# Severity configuration
# ---------------------------------------------------------------------------

@dataclass
class Severity:
    """
    Corruption parameters.

    HONESTY NOTE — put this in your README's Limitations section verbatim:
    these numbers were chosen by eye to look like plausible phone captures.
    They were NOT calibrated against a measured distribution of real-world
    document photos. A principled version would sample from statistics of an
    actual capture dataset. Say so before someone asks.

    Do NOT tune these upward until the degraded split shows the drop you were
    hoping for. That is fitting the experiment to the conclusion, and it is
    exactly the failure mode this project is meant to demonstrate you avoid.
    """

    jpeg_quality: int = 30           # 1-95; lower = more blocking artefacts
    blur_sigma: float = 1.5          # Gaussian radius in pixels
    max_rotation_deg: float = 15.0   # rotation sampled from ±this
    glare_count: int = 1             # elliptical overexposure blobs
    glare_area_frac: float = 0.10    # each blob covers ~this fraction of the frame
    glare_strength: float = 0.75     # 0 = invisible, 1 = pure white
    occlusion_area_frac: float = 0.05  # single opaque rectangle
    compose: bool = True             # apply all corruptions, vs. one at random

    def fingerprint(self) -> str:
        """Short hash so results files can record exactly what was applied."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:8]


DEFAULT_SEVERITY = Severity()


# ---------------------------------------------------------------------------
# Individual corruptions
# ---------------------------------------------------------------------------

def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Recompress as JPEG. Simulates messaging-app re-encoding."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur. Simulates focus miss / motion."""
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_rotation(img: Image.Image, degrees: float) -> Image.Image:
    """
    Rotate with edge-replicated padding.

    `expand=True` keeps the whole document in frame — cropping corners would
    remove ground-truth text and turn a robustness test into a different task.
    Fill colour is sampled from the image border so the padding does not look
    like a synthetic black wedge the model could key on.
    """
    arr = np.asarray(img.convert("RGB"))
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    fill = tuple(int(c) for c in border.mean(axis=0))
    return img.convert("RGB").rotate(degrees, resample=Image.BILINEAR, expand=True, fillcolor=fill)


def apply_glare(img: Image.Image, rng: np.random.Generator, sev: Severity) -> Image.Image:
    """
    Elliptical overexposure blobs — simulates overhead light or flash.

    Implemented as a soft-edged additive white mask rather than a hard ellipse,
    so it blows out highlights the way real glare does instead of stamping a
    shape onto the page.
    """
    img = img.convert("RGB")
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for _ in range(sev.glare_count):
        area = sev.glare_area_frac * w * h
        aspect = rng.uniform(0.6, 1.6)
        rx = float(np.sqrt(area * aspect / np.pi))
        ry = float(np.sqrt(area / (aspect * np.pi)))
        cx = rng.uniform(rx, max(rx + 1, w - rx))
        cy = rng.uniform(ry, max(ry + 1, h - ry))
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(w, h) * 0.03))
    m = np.asarray(mask, dtype=np.float32) / 255.0 * sev.glare_strength
    arr = np.asarray(img, dtype=np.float32)
    out = arr + (255.0 - arr) * m[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def apply_occlusion(img: Image.Image, rng: np.random.Generator, sev: Severity) -> Image.Image:
    """
    Opaque rectangle — a thumb, a fold, a shadow edge.

    Placement is uniform over the frame, which means it will sometimes land on
    whitespace and cost nothing, and sometimes destroy the total. That variance
    is realistic; do not "fix" it by always covering text.
    """
    img = img.convert("RGB")
    w, h = img.size
    area = sev.occlusion_area_frac * w * h
    aspect = rng.uniform(0.4, 2.5)
    rw = min(w - 1, max(4, int(np.sqrt(area * aspect))))
    rh = min(h - 1, max(4, int(np.sqrt(area / aspect))))
    x = int(rng.integers(0, max(1, w - rw)))
    y = int(rng.integers(0, max(1, h - rh)))

    grey = int(rng.integers(30, 90))
    out = img.copy()
    ImageDraw.Draw(out).rectangle([x, y, x + rw, y + rh], fill=(grey, grey, grey))
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

CORRUPTIONS = ["rotation", "glare", "occlusion", "blur", "jpeg"]


def degrade_image(img: Image.Image, seed: int, sev: Severity = DEFAULT_SEVERITY) -> tuple[Image.Image, dict]:
    """
    Apply the corruption pipeline. Returns (image, record-of-what-was-applied).

    ORDER MATTERS and is deliberate: geometry first, then lighting, then
    occlusion, then optical blur, then compression last — mirroring the real
    physical chain (skewed capture → lighting → obstruction → lens → codec).
    Applying JPEG before blur would smooth away the very artefacts it creates.
    """
    rng = np.random.default_rng(seed)
    applied: dict[str, object] = {}

    if sev.compose:
        selected = CORRUPTIONS
    else:
        selected = [CORRUPTIONS[int(rng.integers(0, len(CORRUPTIONS)))]]

    if "rotation" in selected:
        deg = float(rng.uniform(-sev.max_rotation_deg, sev.max_rotation_deg))
        img = apply_rotation(img, deg)
        applied["rotation_deg"] = round(deg, 2)

    if "glare" in selected:
        img = apply_glare(img, rng, sev)
        applied["glare"] = True

    if "occlusion" in selected:
        img = apply_occlusion(img, rng, sev)
        applied["occlusion"] = True

    if "blur" in selected:
        img = apply_blur(img, sev.blur_sigma)
        applied["blur_sigma"] = sev.blur_sigma

    if "jpeg" in selected:
        img = apply_jpeg(img, sev.jpeg_quality)
        applied["jpeg_quality"] = sev.jpeg_quality

    return img, applied


def stable_seed(name: str, global_seed: int) -> int:
    """Per-image seed derived from filename, so output is order-independent."""
    digest = hashlib.sha1(f"{global_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def side_by_side(clean: Image.Image, dirty: Image.Image) -> Image.Image:
    """Stitch clean|degraded for the README figure."""
    h = max(clean.height, dirty.height)
    def fit(im): return im.resize((int(im.width * h / im.height), h), Image.LANCZOS)
    a, b = fit(clean.convert("RGB")), fit(dirty.convert("RGB"))
    canvas = Image.new("RGB", (a.width + b.width + 12, h), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + 12, 0))
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description="Build the degraded test split.")
    p.add_argument("--input", required=True, type=Path, help="directory of clean images")
    p.add_argument("--output", required=True, type=Path, help="directory for degraded images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="process only the first N (for eyeballing)")
    p.add_argument("--save-comparison", action="store_true", help="also write clean|degraded pairs")
    p.add_argument("--jpeg-quality", type=int, default=DEFAULT_SEVERITY.jpeg_quality)
    p.add_argument("--blur-sigma", type=float, default=DEFAULT_SEVERITY.blur_sigma)
    p.add_argument("--max-rotation", type=float, default=DEFAULT_SEVERITY.max_rotation_deg)
    p.add_argument("--independent", action="store_true",
                   help="apply ONE random corruption instead of composing all of them")
    args = p.parse_args()

    sev = Severity(
        jpeg_quality=args.jpeg_quality,
        blur_sigma=args.blur_sigma,
        max_rotation_deg=args.max_rotation,
        compose=not args.independent,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    comparison_dir = args.output.parent / "comparison"
    if args.save_comparison:
        comparison_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    paths = sorted(q for q in args.input.iterdir() if q.suffix.lower() in exts)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no images found in {args.input}")

    manifest = {"seed": args.seed, "severity": asdict(sev),
                "fingerprint": sev.fingerprint(), "images": {}}

    for path in paths:
        clean = Image.open(path).convert("RGB")
        dirty, applied = degrade_image(clean, stable_seed(path.name, args.seed), sev)
        dirty.save(args.output / path.name, quality=95)
        manifest["images"][path.name] = applied
        if args.save_comparison:
            side_by_side(clean, dirty).save(comparison_dir / f"cmp_{path.stem}.jpg", quality=92)

    (args.output.parent / "degradation_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"degraded {len(paths)} images -> {args.output}")
    print(f"severity fingerprint: {sev.fingerprint()}  (record this in results)")
    print(f"mode: {'composed' if sev.compose else 'independent (one random corruption)'}")
    if args.save_comparison:
        print(f"comparisons -> {comparison_dir}  ← LOOK AT THESE before trusting any number")


if __name__ == "__main__":
    main()
