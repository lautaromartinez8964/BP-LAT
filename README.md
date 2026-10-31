# BP-LAT: Boundary-Profile Guided Hybrid Local Adaptive Thresholding

BP-LAT extracts dark targets such as open water from a single-band SAR image. It combines ordered boundary-profile thresholds with a fixed three-component statistical model and entropy-guided fusion. This repository packages the current Q40 + D3E5 + ordered profiles + 0.15 contrast gate + arm 2 + left-tail posterior correction research workflow for independent use.

Prediction uses the SAR raster and its validity mask only; it does not read reference labels, manual regions, optical data or other dates. This is a single-date dark-target mask, not a flood-change map by itself.

## Installation

Install Python 3.10 or newer and create a virtual environment:

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux), then run:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

The verified environment used Python 3.12.13, NumPy 2.5.2, SciPy 1.18.0, scikit-image 0.26.0, scikit-learn 1.9.0 and Rasterio 1.5.1. Other dependency versions can change numerical results. For reproducible experiments use `python -m pip install -r requirements-verified.txt` before installing the package.

## Usage

Run on a single-band, georeferenced raster (GeoTIFF is recommended):

```bash
# Linear power or amplitude DN, with all valid values strictly positive
bp-lat --input /path/to/sar.tif --output-dir /path/to/result --representation positive

# A product already supplied in dB, including negative values
bp-lat --input /path/to/sar_db.tif --output-dir /path/to/result --representation db
```

On Windows, paths containing spaces should be quoted. `python -m bplat.cli` is an equivalent entry point. The representation must be specified by the user; it is not inferred from whether pixel values happen to be positive. Supply a meaningful GeoTIFF NoData value or internal validity mask. For the `positive` branch, any nonpositive pixel that the raster reports as valid causes an error; mark invalid data as NoData first. The `db` branch preserves finite zero and negative values when they are valid.

Use an orthorectified raster with a meaningful, nonzero pixel size and preferably a projected CRS with square pixels. The entire image is loaded into memory, and intermediate arrays and profile records require additional RAM; large scenes may take several minutes or more. The algorithm derives its support radius from the valid scene size, so cropping or tiling a scene changes the method and is not equivalent to a full-scene run. When no source passes the profile and contrast rules, the command reports an error rather than inventing a threshold.

## Output

All outputs retain the input raster grid and CRS:

| File | Meaning |
| --- | --- |
| `mask.tif` | Final mask: 1 dark/water candidate, 0 other valid pixel, 255 NoData. |
| `seed.tif`, `core.tif`, `edges.tif` | Q40 seed, D3E5 core, retained candidate edge pixels. |
| `support.tif` | Pixels with nonzero accepted-source support strength. |
| `anchor.tif`, `threshold.tif` | Geometry-supported anchor and final fused threshold, in the supplied input units. |
| `prior.tif` | Iterated local dark-class prior. |
| `run.json` | Input-file SHA-256, software versions, fitted parameters, crossing, source counts, prior iteration diagnostics and output counts. |

The boolean diagnostic rasters also use 0/1 and 255 NoData. Float outputs use NaN for NoData. Other than `mask.tif`, these rasters are diagnostics, not independent final predictions.

For inputs with sidecar files or external masks, the recorded hash covers only the named input file; archive the sidecars and mask separately for reproducibility.

## Method

1. Use the observed 40th percentile of valid values as a geometry seed and build the core using a 3 x 3 dilation and a 5 x 5 erosion. The seed is not added directly to the final mask.
2. Extract connected Canny edges. Buffer each retained edge by the adaptive scene radius; only oversized sources are recursively split. The same edge buffer supplies the geometry for measurement and spatial support.
3. Measure eight primary short-axis profiles with 50 positions and a cross-strip width derived from 1/16 of the short side. Use one-sided normal profiles as the fallback. Ordered core, transition and bank bands supply each profile's threshold in the native input representation.
4. Require at least two ordered profiles per source; use the observed median threshold, the fraction of passing profiles, scene-rank contrast and bank-sidedness as evidence. Apply the `D >= 0.15` contrast gate and prune nested boxes.
5. Fit exactly three components using deterministic equal-mass samples. Positive linear power or amplitude uses a log-normal likelihood; delivered dB uses a Gaussian likelihood in the supplied dB coordinate. Inputs and final decisions are never globally converted between representations.
6. Identify the qualifying weighted darkest/middle-component left crossing. At each local-prior update, set the dark posterior below that crossing to 1. When no qualifying crossing exists, there is no override.
7. Compute the local equal-cost mixture threshold, floor it at the dark-component mean where geometric support is positive (arm 2), and fuse it with the geometry field using the entropy-derived coefficient.
8. Classify with `input <= threshold`, replace isolated foreground/background pixels synchronously, then remove 8-connected foreground components containing at most four pixels.

The left-tail override changes the posterior used by the local-prior iteration. It does not refit the mixture, modify the mixture risk lookup, or directly add pixels to the final mask.

## Verification and limitations

This package has been checked against the original 2026-10-30 research implementation on all five study inputs in the originating workspace: delivered-dB Da Nang, linear-power Assam, Poyang Lake and Mozambique AOI02, and amplitude-DN Zhengzhou Southwest. Their final masks match pixel for pixel; fitted parameters, accepted source counts, left crossings, profile radii and prior iterations also match. These are implementation equivalence checks on five scenes, not independent accuracy validation or a claim of identical results for every sensor or environment. See `VERIFICATION.md` for the record.

Q40 initialization can be unreliable when dark targets are very sparse or confused with dark land. Results are scene- and representation-dependent: native input does not imply invariance between power, amplitude and dB representations. Parameters and thresholds are estimated separately for each scene. References used during research may include manual and prediction-assisted revisions; they must not be described as independent blind labels without separate evidence.

Run the included checks with:

```bash
python -m unittest discover -s tests -v
```

The repository does not redistribute the original SAR inputs or reference masks. The code is available under the MIT License; see `LICENSE`. `PROVENANCE.md` describes the code's origin.

For a stage-by-stage guide to the functions and parameters, see `METHOD.md`.

## 中文说明

运行时选择原始影像本身的表示：线性功率或幅度 DN 使用 `--representation positive`，产品原有 dB 使用 `--representation db`。岘港影像本身以 dB 供给时，dB 就是该景的 native 输入，不需要再做转换。输出的 `mask.tif` 中 1 为暗目标候选、0 为其他有效像元、255 为无效像元。输出是单时相水体候选，不能直接等同于洪水变化范围。
