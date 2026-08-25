"""Convert a NIfTI MRI volume (.nii / .nii.gz) into 2D PNG slices that
app.py can classify/segment.

Runs on the local computer that receives the scan (a workstation on the
scanner's network), NOT on the Raspberry Pi - the Pi only ever receives
already-converted PNG/JPG slices and runs the app's own inference
pipeline on them. That's why nibabel is in requirements.txt but
deliberately left out of requirements-pi.txt.

Covers the "Receive Scan -> Remove Personal Details -> Prepare the Image"
stages before the app's own "Standardise the Image -> Send to AI Model"
steps take over (on the Pi, over the network, from the PNGs this script
writes).

Important: NIfTI headers themselves rarely carry patient-identifying
fields the way DICOM does (name, ID, DOB usually live in the DICOM series
a .nii was converted from, or in filenames/paths around it). This script
enforces the anonymization that IS its responsibility - output filenames
are sequential case/slice IDs, never derived from the input filename or
path - and flags any non-empty header text field for you to review. It is
NOT a DICOM de-identifier; if your source is DICOM, anonymize at that
stage (e.g. `dcm2niix -ba y`) before this script ever sees a filename.

Usage:
    python nii_to_slices.py --input scan.nii.gz --output slices/
    python nii_to_slices.py --input-dir incoming/ --output slices/ --plane axial
    python nii_to_slices.py --input scan.nii.gz --output slices/ --plane all
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

PLANE_AXES = {"axial": 2, "coronal": 1, "sagittal": 0}
PLANE_CODES = {"axial": "ax", "coronal": "co", "sagittal": "sa"}


def check_header_for_text(img, source_label):
    """NIfTI's free-text fields (descrip, aux_file) aren't where PHI usually
    lives, but flag anything non-empty so a human decides whether to look
    closer - never trust silence here.
    """
    header = img.header
    for field in ("descrip", "aux_file"):
        raw = bytes(header.get(field, b"")).split(b"\x00", 1)[0]
        if raw.strip():
            print(f"  [review] {source_label}: header field '{field}' is non-empty: {raw!r}", file=sys.stderr)


def normalize_slice(slice_2d, low_pct=0.5, high_pct=99.5):
    """Clip to a percentile range (avoids a few hot/dead pixels blowing out
    the scale) then min-max stretch to uint8, matching how the BRISC JPEGs
    are already windowed.
    """
    finite = slice_2d[np.isfinite(slice_2d)]
    if finite.size == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    lo, hi = np.percentile(finite, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    clipped = np.clip(slice_2d, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def is_mostly_empty(slice_u8, min_foreground_frac):
    foreground = (slice_u8 > 10).mean()  # near-black pixels = outside the head
    return foreground < min_foreground_frac


def extract_slices(nii_path, out_dir, case_id, plane, min_foreground_frac, resize):
    img = nib.load(str(nii_path))
    check_header_for_text(img, case_id)

    volume = img.get_fdata()
    if volume.ndim == 4:
        volume = volume[..., 0]  # first volume of a 4D series (e.g. a time series)
    if volume.ndim != 3:
        raise ValueError(f"{nii_path}: expected a 3D (or 4D) volume, got shape {volume.shape}")

    planes = list(PLANE_AXES) if plane == "all" else [plane]
    written = 0
    for plane_name in planes:
        axis = PLANE_AXES[plane_name]
        code = PLANE_CODES[plane_name]
        n_slices = volume.shape[axis]

        for idx in range(n_slices):
            slice_2d = np.take(volume, idx, axis=axis)
            slice_u8 = normalize_slice(slice_2d)
            if is_mostly_empty(slice_u8, min_foreground_frac):
                continue

            image = Image.fromarray(slice_u8, mode="L").convert("RGB")
            if resize:
                image = image.resize((resize, resize), Image.BILINEAR)

            out_name = f"{case_id}_{code}_{idx:04d}.png"
            image.save(out_dir / out_name)
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Single .nii/.nii.gz file")
    src.add_argument("--input-dir", help="Directory of .nii/.nii.gz files to convert")
    parser.add_argument("--output", required=True, help="Directory to write PNG slices into")
    parser.add_argument("--plane", choices=["axial", "coronal", "sagittal", "all"], default="axial",
                         help="Which plane(s) to slice (default: axial, matching BRISC's 'ax' naming)")
    parser.add_argument("--min-foreground", type=float, default=0.02,
                         help="Skip slices with less than this fraction of non-background pixels (default 0.02)")
    parser.add_argument("--resize", type=int, default=None,
                         help="Resize output PNGs to NxN (default: keep native slice resolution; "
                              "app.py resizes to 224/256 itself regardless)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        sources = [Path(args.input)]
    else:
        in_dir = Path(args.input_dir)
        sources = sorted(list(in_dir.glob("*.nii")) + list(in_dir.glob("*.nii.gz")))
        if not sources:
            print(f"No .nii/.nii.gz files found in {in_dir}", file=sys.stderr)
            sys.exit(1)

    total = 0
    for i, path in enumerate(sources, start=1):
        case_id = f"case_{i:04d}"  # never derived from the input filename/path
        print(f"[{i}/{len(sources)}] {path.name} -> {case_id}")
        written = extract_slices(path, out_dir, case_id, args.plane, args.min_foreground, args.resize)
        print(f"  wrote {written} slice(s)")
        total += written

    print(f"\nDone. {total} slice(s) written to {out_dir}/")
    print("Reminder: this only anonymizes the OUTPUT naming and flags header text fields - "
          "if the source was DICOM, anonymize at that conversion step too.")


if __name__ == "__main__":
    main()
