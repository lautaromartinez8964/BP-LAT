# Implementation guide

The command-line entry point is `src/bplat/cli.py`; `run()` in `src/bplat/pipeline.py` coordinates the calculation. It receives the raster array, validity mask, georeferencing profile and an explicit `positive` or `db` representation. The functions in `research_geometry.py` retain the research implementation's numerical and source-geometry rules, including rounding and the use of observed rather than interpolated medians.

## Geometry branch

`build_geometry()` calls `research_geometry.build()`: it takes the observed lower-rank Q40, constructs the D3E5 core and extracts 8-connected Canny edge components. The scene-adaptive radius and minimum edge length are derived from the valid area and short side using `research_config.json`. Each retained source receives a shared Euclidean edge buffer and tight bounding box. Oversized sources are recursively bisected along their longest coordinate axis, then their buffers are reconstructed.

`relative_resample()` replaces each primary profile's transverse strip with the unique odd pixel count `max(3, 2 * floor(short_side / 32) + 1)`. Profile measurements are along the rectangle short axis, at eight locations on its long axis. Normal fallback profiles use the original three-pixel transverse width. `ordinal_estimate()` finds the core and bank bands, searches for ordered boundaries and uses the pooled transition-band samples to estimate a threshold. `_summarize()` keeps every ordered profile and requires at least two. It calculates the source threshold as the observed median and the scene-rank core-to-bank separation. The chosen source passes if that separation is at least 0.15, or if it is undefined. Accepted nested boxes are then pruned.

`anchor_field()` computes the source weight `W = reliability * contrast * sidedness`, with sidedness 1 for bilateral and 0.5 for fallback evidence. It obtains the weighted-median global threshold and combines distance-decayed source contributions. The local support strength is `q = 1 - product(1 - source_weight)`; the geometry anchor shrinks the weighted local threshold towards the global threshold according to q.

## Statistical branch and fusion

`fit_mixture()` uses `ceil(sqrt(N))` midpoint order statistics to fit three Gaussian components with fixed random state and deterministic initialization settings. On positive input the fitting coordinate is log(input); on supplied dB input it is the input itself. The fitted means are sorted, with the darkest component forming the dark class and the other two forming the non-dark class.

`left_tail_crossing()` solves the equality between the globally weighted darkest and middle component densities. A crossing qualifies only if it is to the left of the dark-component mean and the darkest component becomes dominant when moving to the right. In `local_prior()`, every valid pixel below this crossing receives dark posterior 1 during each iteration. No crossing means no override. The local neighborhood radius is derived from the selected profiles, not from a reference mask; the update has the research implementation's half-count regularization and convergence condition.

`lookup()` minimizes the prior-weighted equal-cost classification risk and interpolates the threshold at each local prior. At `q > 0`, the threshold is floored at the dark-component mean in the input representation. `run()` uses binary entropy of the prior to interpolate between the geometry anchor and the mixture threshold. After thresholding, `clean_mask()` synchronously replaces isolated pixels and removes 8-connected foreground components of at most four pixels.

## Source organization

- `pipeline.py`: method orchestration, profile aggregation, model fitting, local prior, lookup and fusion.
- `research_geometry.py`: ported profile sampling, ordering estimator, support geometry and nested-box pruning.
- `research_config.json`: fixed research geometry settings. It is package data and is loaded automatically.
- `numerics.py`: left-tail crossing and final binary cleanup.
- `cli.py`: raster I/O, NoData handling, input hash, software versions and output reports.

See `VERIFICATION.md` for the tested implementation-equivalence cases and their limitations.
