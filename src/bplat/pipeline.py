"""Portable BP-LAT prediction pipeline based on the 2026-10-30 research run.

All prediction inputs are the SAR raster, its validity mask and grid metadata.
No reference data or scene-specific region definitions are used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.special import expit, logsumexp, ndtr
from sklearn.mixture import GaussianMixture

from . import research_geometry as geometry
from .numerics import clean_mask, left_tail_crossing


def _algorithm(pixel_size: float) -> dict:
    """Resolve the frozen geometry parameters using the raster pixel size."""
    raw = json.loads((Path(__file__).parent / "research_config.json").read_text(encoding="utf8"))
    return geometry.resolve_domain_algorithm("linear_power", raw, pixel_size)


def _summarize(raw_profiles: list[dict], scene_values: np.ndarray) -> dict | None:
    """Evaluate each original profile; retain all physically ordered results."""
    valid_profiles = []
    for profile in raw_profiles:
        result = geometry.ordinal_estimate(profile, "ordinal", {"native_pixel_size_m": 1.0})
        result.update(profile_index=profile["profile_index"], kind=profile["kind"],
                      step_pixels=profile["step_pixels"])
        if (result.get("status") == "accepted"
                and np.isfinite(result.get("local_threshold", np.nan))
                and result["core_median"] < result["local_threshold"] < result["bank_median"]):
            result["aggregation_inlier"] = True
            valid_profiles.append(result)
    if len(valid_profiles) < 2:
        return None
    contrasts = []
    for profile in valid_profiles:
        ranks = np.searchsorted(scene_values, [profile["core_median"], profile["bank_median"]], side="right") / len(scene_values)
        contrasts.append(max(0.0, float(ranks[1] - ranks[0])))
    return {"profiles": valid_profiles,
            "threshold": geometry.observed_quantile(np.array([p["local_threshold"] for p in valid_profiles]), 0.5),
            "reliability": len(valid_profiles) / max(1, len(raw_profiles)),
            "scene_contrast": geometry.observed_quantile(np.asarray(contrasts), 0.5)}


def build_geometry(values: np.ndarray, valid: np.ndarray, profile: dict) -> tuple[dict, list, int]:
    """Build the shared edge supports, estimate profiles and prune nested boxes."""
    pixel_size = abs(float(profile["transform"].a))
    if not pixel_size:
        raise ValueError("Raster pixel size must be nonzero")
    algorithm = _algorithm(pixel_size)
    geom = geometry.build(values, valid, profile, algorithm, 40, 5, 3, True)
    geom, _ = geometry.relative_resample(geom, values, valid, algorithm)
    sorted_values = np.sort(values[valid])
    pairs = []
    for source in geom["sources"]:
        primary = _summarize(source["profiles"], sorted_values)
        fallback = _summarize(source["fallback_profiles"], sorted_values)
        selected = primary if primary is not None else fallback
        if selected is None:
            continue
        contrast = selected["scene_contrast"]
        if np.isfinite(contrast) and contrast < 0.15:
            continue
        evidence = "double" if primary is not None else "single"
        pairs.append((source, {"selected": selected, "evidence": evidence}))
    boxes = {s["candidate_id"]: tuple(int(v) for v in s["box"]) for s, _ in pairs}
    removed, _ = geometry.greedy_nested_prune(boxes, None)
    pairs = [(s, r) for s, r in pairs if s["candidate_id"] not in removed]
    if not pairs:
        raise RuntimeError("No valid source passed the BP-LAT profile and contrast gates")
    return geom, pairs, len(removed)


def fit_mixture(values: np.ndarray, valid: np.ndarray, representation: str) -> dict:
    """Fit K=3 on deterministic midpoint order statistics of equal-mass bins."""
    sample = np.sort(values[valid])
    n = int(np.ceil(np.sqrt(sample.size)))
    sample = sample[np.minimum(((np.arange(n) + .5) * sample.size / n).astype(int), sample.size - 1)]
    if representation == "positive":
        if np.any(sample <= 0):
            raise ValueError("Positive representation requires strictly positive valid pixels")
        sample = np.log(sample)
    variance = float(np.var(sample, ddof=1))
    reg = max(np.finfo(float).eps, variance * np.sqrt(np.finfo(float).eps))
    fit = GaussianMixture(3, random_state=0, max_iter=1000, tol=1e-6,
                          n_init=8, init_params="kmeans", reg_covar=reg).fit(sample[:, None])
    if not fit.converged_:
        raise RuntimeError("K=3 mixture did not converge")
    order = np.argsort(fit.means_[:, 0])
    return {"mu": fit.means_[order, 0], "var": fit.covariances_[order, 0, 0],
            "pi": fit.weights_[order], "n_samples": n, "reg_covar": reg}


def anchor_field(shape: tuple[int, int], valid: np.ndarray, pairs: list) -> tuple[np.ndarray, np.ndarray, float]:
    weights = np.array([r["selected"]["reliability"] * r["selected"]["scene_contrast"] *
                        (1.0 if r["evidence"] == "double" else 0.5) for _, r in pairs])
    thresholds = np.array([r["selected"]["threshold"] for _, r in pairs])
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise RuntimeError("Accepted sources have undefined or zero aggregate evidence weights")
    tau = geometry.weighted_median(thresholds, weights)
    numerator, denominator = np.zeros(shape), np.zeros(shape)
    absence = np.ones(shape)
    for (source, _), threshold, weight in zip(pairs, thresholds, weights):
        r0, r1, c0, c1 = source["support_box"]
        window = np.s_[r0:r1, c0:c1]
        local_weight = np.asarray(source["weight"], float) * weight
        numerator[window] += local_weight * threshold
        denominator[window] += local_weight
        absence[window] *= 1 - local_weight
    local = np.full(shape, tau)
    present = denominator > 0
    local[present] = numerator[present] / denominator[present]
    q = 1 - absence
    field = tau + q * (local - tau)
    field[~valid] = np.nan
    return field, q, tau


def _profile_radius(pairs: list) -> float:
    terms = [float(p["step_pixels"]) + float(p.get("band_samples", 0))
             for _, record in pairs for p in record["selected"]["profiles"]]
    return float(np.ceil(np.median(terms)))


def local_prior(values: np.ndarray, valid: np.ndarray, model: dict, radius: float,
                representation: str, tail: np.ndarray) -> tuple[np.ndarray, dict]:
    z = values[valid] if representation == "db" else np.log(values[valid])
    mu, var, weights = model["mu"], model["var"], model["pi"]
    lp = -.5 * (np.log(2 * np.pi * var) + (z[:, None] - mu) ** 2 / var)
    ratio = np.zeros(values.shape, np.float32)
    ratio[valid] = lp[:, 0] - logsumexp(lp[:, 1:] + np.log(weights[1:] / weights[1:].sum()), axis=1)
    size = 2 * max(1, int(np.ceil(radius))) + 1
    observed = ndimage.uniform_filter(valid.astype(float), size, mode="constant") * size ** 2
    prior = np.full(values.shape, weights[0], float)
    for iteration in range(100):
        p = np.clip(prior[valid], 1e-12, 1 - 1e-12)
        posterior = np.zeros(values.shape, np.float32)
        posterior[valid] = expit(ratio[valid] + np.log(p) - np.log1p(-p))
        posterior[tail] = 1.0
        total = ndimage.uniform_filter(posterior, size, mode="constant") * size ** 2
        updated = (total + .5) / (observed + 1.)
        delta = float(np.abs(updated[valid] - prior[valid]).mean())
        prior = updated
        if delta < 1 / np.sqrt(valid.sum()):
            break
    return prior, {"window": size, "iterations": iteration + 1, "delta": delta}


def lookup(values: np.ndarray, valid: np.ndarray, model: dict, representation: str) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the research implementation's equal-cost risk lookup."""
    x = values[valid]
    if representation == "db":
        grid = np.linspace(float(x.min()), float(x.max()), 2048)
        z = grid
    else:
        x = np.sort(x)
        n = int(np.ceil(np.sqrt(x.size)))
        sample = x[np.minimum(((np.arange(n) + .5) * x.size / n).astype(int), x.size - 1)]
        grid = np.unique(np.r_[0, sample, np.max(x)])
        z = np.log(np.maximum(grid, np.finfo(float).tiny))
    mu, var, weights = model["mu"], model["var"], model["pi"]
    cdf = ndtr((z[None, :] - mu[:, None]) / np.sqrt(var[:, None]))
    bright = (cdf[1:] * (weights[1:] / weights[1:].sum())[:, None]).sum(axis=0)
    probabilities = np.linspace(0, 1, len(grid))
    thresholds = np.array([grid[np.argmin(p * (1 - cdf[0]) + (1 - p) * bright)] for p in probabilities])
    return probabilities, np.maximum.accumulate(thresholds)


def run(values: np.ndarray, valid: np.ndarray, profile: dict, representation: str) -> tuple[dict, dict]:
    """Execute the locked BP-LAT workflow on one raster."""
    if representation not in ("positive", "db"):
        raise ValueError("representation must be positive or db")
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or valid.shape != values.shape or valid.sum() < 100:
        raise ValueError("Input needs a 2-D raster with at least 100 valid pixels")
    if representation == "positive" and np.any(values[valid] <= 0):
        raise ValueError("Positive input contains non-positive valid values; fix the input validity mask")
    print("Constructing geometry and evaluating profiles...", flush=True)
    geom, pairs, pruned = build_geometry(values, valid, profile)
    print(f"Fitting K=3 and local priors ({len(pairs)} sources)...", flush=True)
    model = fit_mixture(values, valid, representation)
    crossing = left_tail_crossing(model)
    crossing_native = None if crossing is None else float(np.exp(crossing) if representation == "positive" else crossing)
    tail = valid & (values < crossing_native) if crossing_native is not None else np.zeros(values.shape, bool)
    anchor, q, tau = anchor_field(values.shape, valid, pairs)
    radius = _profile_radius(pairs)
    prior, diag = local_prior(values, valid, model, radius, representation, tail)
    probabilities, cuts = lookup(values, valid, model, representation)
    tmix = np.interp(prior, probabilities, cuts)
    dark_mean = float(np.exp(model["mu"][0] + model["var"][0] / 2) if representation == "positive" else model["mu"][0])
    p = np.clip(prior, np.finfo(float).eps, 1 - np.finfo(float).eps)
    certainty = 1 + (p * np.log(p) + (1 - p) * np.log1p(-p)) / np.log(2)
    threshold = anchor + certainty * (np.where(q > 0, np.maximum(tmix, dark_mean), tmix) - anchor)
    mask = clean_mask(valid & (values <= threshold), valid, 5)
    support = q > 0
    arrays = {"mask": mask, "seed": geom["seed"], "core": geom["core"], "edges": geom["edges"],
              "support": support, "anchor": anchor, "threshold": threshold, "prior": prior}
    meta = {"representation": representation, "model": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()},
            "crossing_native": crossing_native, "tail_pixels": int(tail.sum()), "dark_mean_native": dark_mean,
            "candidate_sources": len(geom["sources"]), "accepted_sources": len(pairs), "nested_pruned": pruned,
            "tau_star": tau, "profile_radius_pixels": radius, "prior": diag, "water_pixels": int(mask.sum())}
    return arrays, meta
