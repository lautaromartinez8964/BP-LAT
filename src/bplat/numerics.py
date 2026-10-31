"""Crossing and binary cleanup routines."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy import ndimage as ndi

def left_tail_crossing(model: dict[str, Any]) -> float | None:
    """Return qualifying left crossing of weighted darkest/middle densities."""
    mu, var, weight = model["mu"], model["var"], model["pi"]
    a = -0.5 / var[0] + 0.5 / var[1]
    b = mu[0] / var[0] - mu[1] / var[1]
    c = (np.log(weight[0] / weight[1]) - 0.5 * np.log(var[0] / var[1])
         - mu[0] ** 2 / (2 * var[0]) + mu[1] ** 2 / (2 * var[1]))
    roots = np.roots([a, b, c]) if abs(a) > 1e-14 else np.roots([b, c])
    valid_roots = sorted(float(root.real) for root in roots
                         if abs(root.imag) < 1e-9 and root.real < mu[0])
    for root in reversed(valid_roots):
        delta = max(1e-6, np.sqrt(var[0]) * 1e-4)
        polynomial = lambda x: (a * x + b) * x + c
        if polynomial(root - delta) < 0 < polynomial(root + delta):
            return root
    return None


def clean_mask(mask: np.ndarray, valid: np.ndarray, minimum_size: int) -> np.ndarray:
    """Apply 8-connected foreground cleanup and remove tiny water objects."""
    structure = np.ones((3, 3), dtype=bool)
    raw = mask & valid
    kernel = structure.astype(np.uint8)
    kernel[1, 1] = 0
    valid_neighbors = ndi.convolve(valid.astype(np.int16), kernel, mode="constant")
    foreground_neighbors = ndi.convolve(raw.astype(np.int16), kernel, mode="constant")
    same = np.where(raw, foreground_neighbors, valid_neighbors - foreground_neighbors)
    isolated = valid & (valid_neighbors > 0) & (same == 0)
    cleaned = raw.copy()
    cleaned[isolated] = foreground_neighbors[isolated] * 2 > valid_neighbors[isolated]
    labels, count = ndi.label(cleaned, structure=structure)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= minimum_size
    keep[0] = False
    return keep[labels] & valid
