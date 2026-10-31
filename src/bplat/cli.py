"""Command-line interface for one GeoTIFF input."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import rasterio

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the readable BP-LAT reference pipeline")
    parser.add_argument("--input", required=True, type=Path, help="single-band SAR GeoTIFF")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--representation", required=True, choices=("positive", "db"),
                        help="positive = linear power/amplitude, db = supplied dB")
    args = parser.parse_args()

    output_names = ("mask", "seed", "core", "edges", "support", "anchor", "threshold", "prior")
    if args.input.resolve() in {(args.output_dir / f"{name}.tif").resolve() for name in output_names}:
        raise ValueError("Input cannot be one of the output files")

    with rasterio.open(args.input) as dataset:
        if dataset.count != 1:
            raise ValueError("Input must contain exactly one raster band")
        values = dataset.read(1).astype(np.float64)
        valid = (dataset.read_masks(1) > 0) & np.isfinite(values)
        profile = dataset.profile.copy()

    arrays, metadata = run(values, valid, profile, args.representation)
    digest = hashlib.sha256()
    with args.input.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata["input_sha256"] = digest.hexdigest()
    metadata["software_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "scikit-image", "scikit-learn", "rasterio")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        output_profile = {"driver": "GTiff", "height": values.shape[0], "width": values.shape[1],
                          "crs": profile.get("crs"), "transform": profile["transform"],
                          "dtype": "uint8" if array.dtype == bool else "float32",
                          "count": 1, "compress": "deflate",
                          "nodata": 255 if array.dtype == bool else np.nan}
        output = np.asarray(array, dtype=output_profile["dtype"])
        output[~valid] = output_profile["nodata"]
        with rasterio.open(args.output_dir / f"{name}.tif", "w", **output_profile) as dst:
            dst.write(output, 1)
            dst.write_mask(valid.astype(np.uint8) * 255)
    (args.output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'mask.tif'} ({metadata['accepted_sources']} sources)")


if __name__ == "__main__":
    main()
