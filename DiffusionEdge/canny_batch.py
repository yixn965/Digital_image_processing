#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Auto threshold selection using the image median (common standard practice).
    lower = (1-sigma)*median, upper=(1+sigma)*median
    """
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        paths = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(paths)

def main():
    ap = argparse.ArgumentParser(description="Batch Canny edge detection with standard preprocessing.")
    ap.add_argument("--input_dir", required=True, type=str, help="Folder containing input images")
    ap.add_argument("--output_dir", required=True, type=str, help="Folder to save edge images")
    ap.add_argument("--recursive", action="store_true", help="Recursively process subfolders")
    ap.add_argument("--blur_ksize", type=int, default=5, help="Gaussian blur kernel size (odd int). default=5")
    ap.add_argument("--blur_sigma", type=float, default=1.0, help="Gaussian blur sigma. default=1.0")
    ap.add_argument("--auto", action="store_true", help="Use auto-canny thresholds (recommended)")
    ap.add_argument("--auto_sigma", type=float, default=0.33, help="Auto-canny sigma. default=0.33")
    ap.add_argument("--t1", type=int, default=100, help="Canny threshold1 (used when --auto is not set)")
    ap.add_argument("--t2", type=int, default=200, help="Canny threshold2 (used when --auto is not set)")
    ap.add_argument("--suffix", type=str, default="_canny", help="Suffix added before extension. default=_canny")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"input_dir not found or not a directory: {in_dir}")

    if args.blur_ksize <= 0 or args.blur_ksize % 2 == 0:
        raise SystemExit("--blur_ksize must be a positive odd integer (e.g., 3,5,7).")

    img_paths = collect_images(in_dir, args.recursive)
    if not img_paths:
        raise SystemExit(f"No images found under {in_dir}")

    processed = 0
    for p in img_paths:
        # Load (BGR)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read: {p}")
            continue

        # Standard preprocessing: grayscale -> Gaussian blur -> Canny
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (args.blur_ksize, args.blur_ksize), args.blur_sigma)

        if args.auto:
            edges = auto_canny(gray_blur, sigma=args.auto_sigma)
        else:
            edges = cv2.Canny(gray_blur, args.t1, args.t2)

        # Output path: preserve relative structure
        rel = p.relative_to(in_dir)
        save_parent = (out_dir / rel.parent)
        save_parent.mkdir(parents=True, exist_ok=True)

        out_name = f"{p.stem}{args.suffix}.png"  # always save as png
        out_path = save_parent / out_name

        ok = cv2.imwrite(str(out_path), edges)
        if not ok:
            print(f"[WARN] Failed to write: {out_path}")
            continue

        processed += 1

    print(f"Done. Processed {processed}/{len(img_paths)} images.")
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()
