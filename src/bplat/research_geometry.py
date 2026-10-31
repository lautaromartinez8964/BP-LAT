"""Ported from the project research implementation; see PROVENANCE.md."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.stats import rankdata
from skimage.feature import canny


def adaptive_geometry(
    valid: np.ndarray,
    pixel_size_m: float,
    algorithm: dict[str, Any],
    *,
    scene_fraction_key: str = "edge_buffer_scene_fraction",
    maximum_short_side_fraction_key: str = "edge_buffer_maximum_short_side_fraction",
    transition_fraction_key: str = "transition_buffer_fraction",
) -> dict[str, float | int]:
    short_side_pixels = int(min(valid.shape))
    maximum_radius_fraction = float(algorithm[maximum_short_side_fraction_key])
    if not 0.0 < maximum_radius_fraction <= 0.5:
        raise ValueError("The maximum short-side support fraction must lie in (0, 0.5].")
    minimum_short_side = max(3, int(np.ceil(1.0 / maximum_radius_fraction)))
    if short_side_pixels < minimum_short_side:
        raise ValueError(
            f"Scene short side ({short_side_pixels} px) is below the {minimum_short_side} px local-adaptation minimum."
        )
    equivalent_side_pixels = float(np.sqrt(int(valid.sum())))
    nominal_radius = float(algorithm[scene_fraction_key]) * equivalent_side_pixels
    maximum_radius = float(algorithm[maximum_short_side_fraction_key]) * short_side_pixels
    radius_pixels = max(1, int(np.ceil(min(nominal_radius, maximum_radius))))
    transition_pixels = min(
        radius_pixels,
        max(1, int(np.ceil(radius_pixels * float(algorithm[transition_fraction_key])))),
    )
    inner_pixels = max(0, radius_pixels - transition_pixels)
    minimum_edge_pixels = max(
        int(algorithm["minimum_edge_connected_floor_pixels"]),
        int(np.ceil(float(algorithm["minimum_edge_connected_scene_fraction"]) * equivalent_side_pixels)),
    )
    segment_pixels = max(
        minimum_edge_pixels,
        int(np.ceil(float(algorithm["source_segment_scene_fraction"]) * equivalent_side_pixels)),
    )
    overlap_pixels = min(
        segment_pixels - 1,
        int(np.ceil(float(algorithm["source_segment_overlap_scene_fraction"]) * equivalent_side_pixels)),
    )
    return {
        "valid_scene_pixels": int(valid.sum()),
        "equivalent_scene_side_pixels": equivalent_side_pixels,
        "equivalent_scene_side_m": equivalent_side_pixels * pixel_size_m,
        "raster_short_side_pixels": short_side_pixels,
        "minimum_raster_short_side_pixels": minimum_short_side,
        "native_pixel_size_m": pixel_size_m,
        "support_radius_pixels": radius_pixels,
        "support_radius_effective_m": radius_pixels * pixel_size_m,
        "transition_width_pixels": transition_pixels,
        "transition_width_effective_m": transition_pixels * pixel_size_m,
        "inner_support_radius_pixels": inner_pixels,
        "inner_support_radius_effective_m": inner_pixels * pixel_size_m,
        "minimum_retained_edge_pixels": minimum_edge_pixels,
        "source_segment_pixels": segment_pixels,
        "source_overlap_pixels": overlap_pixels,
        "support_radius_scene_fraction_effective": radius_pixels / equivalent_side_pixels,
        "support_radius_short_side_fraction_effective": radius_pixels / short_side_pixels,
    }


def resolve_domain_algorithm(domain: str, algorithm: dict[str, Any], pixel_size_m: float) -> dict[str, Any]:
    resolved = dict(algorithm)
    resolved.setdefault("global_backbone_aggregation", "source_median")
    resolved.setdefault("maximum_accepted_support_scene_fraction", 0.75)
    resolved.setdefault("allocation_edge_buffer_scene_fraction", resolved["edge_buffer_scene_fraction"])
    resolved.setdefault("allocation_edge_buffer_maximum_short_side_fraction", resolved["edge_buffer_maximum_short_side_fraction"])
    resolved.setdefault("allocation_transition_buffer_fraction", resolved["transition_buffer_fraction"])
    resolved.setdefault("allocation_anchor_full_e5_component", False)
    resolved.setdefault("allocation_anchor_max_component_scene_fraction", 0.0)
    resolved.setdefault("edge_normal_fallback_enabled", False)
    resolved.setdefault("fallback_profile_count", resolved["local_profile_count"])
    resolved.setdefault("fallback_minimum_passed_profiles", resolved["local_profile_minimum_passed"])
    resolved.setdefault("fallback_tangent_radius_pixels", 6.0)
    resolved.setdefault("fallback_normal_half_width_pixels", 18)
    resolved.setdefault("fallback_cross_tangent_half_width_pixels", 1)
    resolved.setdefault("fallback_core_near_edge_pixels", 4)
    resolved.setdefault("fallback_bank_gap_pixels", resolved["profile_bank_gap_samples"])
    resolved.setdefault("fallback_bank_width_pixels", resolved["profile_bank_width_samples"])
    resolved.setdefault("backbone_use_edge_normal_fallback", False)
    resolved.setdefault("proximity_monotone_relaxation", False)
    resolved.setdefault("fallback_evidence_priority", False)
    for name in (
        "profile_bank_minimum_relative_contrast",
        "local_profile_minimum_passed",
        "profile_threshold_max_relative_mad",
        "profile_cross_statistic",
        "global_backbone_aggregation",
    ):
        if isinstance(resolved[name], dict):
            resolved[name] = resolved[name][domain]
    if isinstance(resolved["fallback_minimum_passed_profiles"], dict):
        resolved["fallback_minimum_passed_profiles"] = resolved["fallback_minimum_passed_profiles"][domain]
    resolved["local_profile_gradient_min_search_m"] = (
        float(resolved["local_profile_gradient_min_search_pixels"]) * pixel_size_m
    )
    resolved["local_profile_gradient_max_search_m"] = (
        float(resolved["local_profile_gradient_max_search_pixels"]) * pixel_size_m
    )
    resolved["native_pixel_size_m"] = pixel_size_m
    return resolved


def source_weight(distance_pixels: np.ndarray, geometry: dict[str, float | int]) -> np.ndarray:
    radius = float(geometry["support_radius_pixels"])
    inner = float(geometry["inner_support_radius_pixels"])
    transition = max(radius - inner, 1.0)
    weight = np.clip((radius - distance_pixels) / transition, 0.0, 1.0)
    weight[distance_pixels <= inner] = 1.0
    return weight


def principal_axis(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    centered = points.astype(np.float64) - points.mean(axis=0)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.linalg.norm(direction))
    if norm <= np.finfo(float).eps:
        return None
    return direction / norm


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    return [(int(group[0]), int(group[-1]) + 1) for group in np.split(indices, boundaries)]


def observed_quantile(values: np.ndarray, quantile: float) -> float:
    sample = np.sort(np.asarray(values, dtype=float).ravel())
    if not sample.size or not np.all(np.isfinite(sample)):
        raise ValueError("A finite, nonempty sample is required")
    return float(sample[int(np.floor(quantile * (sample.size - 1)))])


def sample_strip(values: np.ndarray, valid: np.ndarray, core: np.ndarray,
                 rows: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inside = (rows >= 0) & (columns >= 0) & (rows < values.shape[0]) & (columns < values.shape[1])
    rows = np.clip(rows, 0, values.shape[0] - 1)
    columns = np.clip(columns, 0, values.shape[1] - 1)
    usable = inside & valid[rows, columns]
    samples = np.where(usable, values[rows, columns], np.nan)
    profile = np.full(rows.shape[0], np.nan)
    fraction = np.zeros(rows.shape[0])
    for index in range(rows.shape[0]):
        if usable[index].any():
            profile[index] = float(np.median(samples[index, usable[index]]))
            fraction[index] = float(np.mean(core[rows[index, usable[index]], columns[index, usable[index]]]))
    return profile, fraction, samples


def rectangle_profiles(values: np.ndarray, valid: np.ndarray, core: np.ndarray,
                       box: tuple[int, int, int, int], algorithm: dict[str, Any]) -> list[dict[str, Any]]:
    row_start, row_stop, column_start, column_stop = box
    height, width = row_stop - row_start, column_stop - column_start
    length = min(height, width)
    cross_width = max(1.0, length * algorithm["profile_cross_width_fraction"])
    sample_count = int(algorithm["profile_samples"])
    cross_count = max(3, int(np.ceil(cross_width)) + 1)
    records = []
    for index, fraction in enumerate(np.linspace(0.1, 0.9, int(algorithm["local_profile_count"]))):
        if height <= width:
            positions = np.linspace(row_start, row_stop - 1, sample_count)
            center = column_start + fraction * max(width - 1, 1)
            cross = np.linspace(center - cross_width / 2, center + cross_width / 2, cross_count)
            rows, columns = np.meshgrid(positions, cross, indexing="ij")
            axis = "vertical_short_axis"
        else:
            positions = np.linspace(column_start, column_stop - 1, sample_count)
            center = row_start + fraction * max(height - 1, 1)
            cross = np.linspace(center - cross_width / 2, center + cross_width / 2, cross_count)
            columns, rows = np.meshgrid(positions, cross, indexing="ij")
            axis = "horizontal_short_axis"
        rows = np.clip(np.rint(rows).astype(int), 0, values.shape[0] - 1)
        columns = np.clip(np.rint(columns).astype(int), 0, values.shape[1] - 1)
        profile, core_fraction, samples = sample_strip(values, valid, core, rows, columns)
        records.append({"profile_index": index, "kind": "double", "axis": axis,
                        "profile": profile, "core": core_fraction >= algorithm["profile_core_presence_fraction"],
                        "samples": samples, "step_pixels": (length - 1) / (sample_count - 1),
                        "centers_rc": np.column_stack((rows.mean(axis=1), columns.mean(axis=1)))})
    return records


def normal_profiles(values: np.ndarray, valid: np.ndarray, core: np.ndarray,
                    segment: np.ndarray, algorithm: dict[str, Any]) -> list[dict[str, Any]]:
    points = np.argwhere(segment).astype(float)
    global_tangent = principal_axis(points)
    if global_tangent is None:
        return []
    projections = (points - points.mean(axis=0)) @ global_tangent
    lower, upper = projections.min(), projections.max()
    positions = np.linspace(lower + 0.1 * (upper - lower), upper - 0.1 * (upper - lower), int(algorithm["fallback_profile_count"]))
    half_width = int(algorithm["fallback_normal_half_width_pixels"])
    offsets = np.arange(-half_width, half_width + 1, dtype=float)
    tangent_band = int(algorithm["fallback_cross_tangent_half_width_pixels"])
    cross = np.arange(-tangent_band, tangent_band + 1, dtype=float)
    records = []
    for index, position in enumerate(positions):
        anchor = points[np.argmin(np.abs(projections - position))]
        nearby = points[np.linalg.norm(points - anchor, axis=1) <= algorithm["fallback_tangent_radius_pixels"]]
        tangent = principal_axis(nearby)
        if tangent is None:
            tangent = global_tangent
        normal = np.array([-tangent[1], tangent[0]])
        centers = anchor + offsets[:, None] * normal
        rows = np.rint(centers[:, 0, None] + cross * tangent[0]).astype(int)
        columns = np.rint(centers[:, 1, None] + cross * tangent[1]).astype(int)
        profile, fraction, samples = sample_strip(values, valid, core, rows, columns)
        records.append({"profile_index": index, "kind": "single", "profile": profile,
                        "core": fraction > 0, "samples": samples, "step_pixels": 1.0,
                        "centers_rc": centers, "zero_index": half_width})
    return records


def equal_error_threshold(core_samples: np.ndarray, bank_samples: np.ndarray) -> tuple[float, float]:
    core_values = np.sort(core_samples[np.isfinite(core_samples)])
    bank_values = np.sort(bank_samples[np.isfinite(bank_samples)])
    if not core_values.size or not bank_values.size:
        raise ValueError("Both class samples are required")
    candidates = np.unique(np.concatenate((core_values, bank_values)))
    false_negative = 1.0 - np.searchsorted(core_values, candidates, side="right") / core_values.size
    false_positive = np.searchsorted(bank_values, candidates, side="right") / bank_values.size
    score = np.maximum(false_negative, false_positive)
    index = int(np.argmin(score))
    return float(candidates[index]), float(score[index])


def profile_run(record: dict[str, Any]) -> tuple[int, int] | None:
    profile = record["profile"]
    runs = contiguous_runs(record["core"] & np.isfinite(profile))
    if not runs:
        return None
    if record["kind"] == "double":
        return max(runs, key=lambda run: run[1] - run[0])
    zero = record["zero_index"]
    run = min(runs, key=lambda item: np.min(np.abs(np.arange(*item) - zero)))
    if min(abs(np.arange(*run) - zero)) > 4 or run[0] <= zero < run[1]:
        return None
    return run


def ordinal_estimate(record: dict[str, Any], estimator: str, algorithm: dict[str, Any]) -> dict[str, Any]:
    profile = np.array([observed_quantile(samples[np.isfinite(samples)], .5) if np.any(np.isfinite(samples)) else np.nan for samples in record["samples"]])
    run = profile_run(record)
    if run is None or not np.all(np.isfinite(profile)):
        return {"status": "no_finite_core_profile"}
    start, stop = run
    step = max(float(record["step_pixels"]), np.finfo(float).eps)
    width = max(3, int(np.ceil(min((stop-start)/4, 6/step))))
    gap = max(1, int(np.ceil(1.5/step)))
    search = max(width+gap, int(np.ceil(15/step)))
    core_indices = np.arange(start, stop)
    if core_indices.size < 3:
        return {"status": "core_band_extent"}
    directions = [-1, 1] if record["kind"] == "double" else ([1] if stop <= record["zero_index"] else [-1])
    bank_bands, inner_bands, boundaries, ordering_scores = [], [], [], []
    for direction in directions:
        origin = start if direction < 0 else stop-1
        positions = origin + direction*np.arange(1, search+1)
        choices = []
        for boundary in positions:
            inner = boundary-direction*np.arange(gap, gap+width)
            outer = boundary+direction*np.arange(gap, gap+width)
            if min(inner.min(), outer.min()) < 0 or max(inner.max(), outer.max()) >= profile.size:
                continue
            core_value = observed_quantile(profile[core_indices], .5)
            inner_value = observed_quantile(profile[inner], .5)
            outer_value = observed_quantile(profile[outer], .5)
            if not outer_value > inner_value or not outer_value > core_value:
                continue
            ordinal_gradient = float(np.mean(profile[outer, None] > profile[inner][None, :]) - np.mean(profile[outer, None] < profile[inner][None, :]))
            if ordinal_gradient <= 0:
                continue
            persistence = float(np.mean(profile[outer] > inner_value))
            choices.append((ordinal_gradient*persistence, int(boundary), inner, outer))
        if not choices:
            return {"status": "no_positive_ordinal_gradient"}
        scores = np.array([choice[0] for choice in choices])
        maximum = scores.max()
        best_indices = np.flatnonzero(scores == maximum)
        choice = choices[int(best_indices[(len(best_indices)-1)//2])]
        boundaries.append(choice[1])
        inner_bands.append(choice[2])
        bank_bands.append(choice[3])
        ordering_scores.append(choice[0])
    transition = np.unique(np.concatenate([np.arange(boundary-gap, boundary+gap+1) for boundary in boundaries]))
    core_samples = record["samples"][core_indices].ravel()
    bank_samples = np.concatenate([record["samples"][indices].ravel() for indices in bank_bands])
    inner_samples = np.concatenate([record["samples"][indices].ravel() for indices in inner_bands])
    core_value = observed_quantile(core_samples[np.isfinite(core_samples)], .5)
    bank_value = observed_quantile(bank_samples[np.isfinite(bank_samples)], .5)
    _, separation_error = equal_error_threshold(core_samples, bank_samples)
    samples = record["samples"][transition].ravel()
    threshold = observed_quantile(samples[np.isfinite(samples)], .5)
    if estimator == "ordinal_plateau":
        threshold = (core_value+bank_value)/2
    elif estimator == "ordinal_balanced":
        threshold, _ = equal_error_threshold(inner_samples, bank_samples)
    if not core_value < threshold < bank_value:
        return {"status": "threshold_outside_plateaus", "local_threshold": threshold,
                "core_median": core_value, "bank_median": bank_value}
    return {"status": "accepted", "local_threshold": threshold, "core_median": core_value, "bank_median": bank_value,
            "core_indices": core_indices.tolist(), "bank_indices": [indices.tolist() for indices in bank_bands],
            "boundary_indices": boundaries, "transition_indices": transition.tolist(), "band_samples": width,
            "band_width_m": width*step*algorithm["native_pixel_size_m"], "ordinal_gradient_scores": ordering_scores,
            "class_separation": max(0., 1-2*separation_error)}


def pad_slice(r0: int, r1: int, c0: int, c1: int, pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - pad),
        min(shape[0], r1 + pad),
        max(0, c0 - pad),
        min(shape[1], c1 + pad),
    )


def local_distance_crop(mask_crop: np.ndarray, origin: tuple[int, int], radius: float, shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """EDT of a crop, then re-pad so distances <= radius are exact in global coords."""
    if not mask_crop.any():
        return None
    rows, cols = np.where(mask_crop)
    pad = int(np.ceil(radius)) + 1
    r0, r1, c0, c1 = pad_slice(
        int(rows.min()) + origin[0],
        int(rows.max()) + origin[0] + 1,
        int(cols.min()) + origin[1],
        int(cols.max()) + origin[1] + 1,
        pad,
        shape,
    )
    crop = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    crop[rows + origin[0] - r0, cols + origin[1] - c0] = True
    return (r0, r1, c0, c1), ndimage.distance_transform_edt(~crop)


def crop_bbox(mask_crop: np.ndarray, origin: tuple[int, int]) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask_crop)
    return (
        int(rows.min()) + origin[0],
        int(rows.max()) + origin[0] + 1,
        int(cols.min()) + origin[1],
        int(cols.max()) + origin[1] + 1,
    )


def window_from_crop(distance: np.ndarray, bounds: tuple[int, int, int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, _, c0, _ = bounds
    return distance[box[0] - r0:box[1] - r0, box[2] - c0:box[3] - c0]


def components(core,v,a):
    e=canny(core.astype(np.float32),sigma=a['canny_sigma'])&v
    labs,n=ndimage.label(e,structure=np.ones((3,3)));sizes=np.bincount(labs.ravel())
    return e,labs,sizes,ndimage.find_objects(labs)


def split_points(points,r,limit):
    """Bisect oversized edge point sets along longest axis; no duplicated edge pixel."""
    pending=[points];out=[]
    while pending:
        pt=pending.pop();span=np.ptp(pt,axis=0)+1
        # Conservative full un-clipped buffer box, guaranteeing the area cap.
        if np.prod(span+2*int(np.ceil(r)))<=limit:
            out.append(pt);continue
        ax=int(np.argmax(span));mid=(pt[:,ax].min()+pt[:,ax].max())//2
        left=pt[pt[:,ax]<=mid];right=pt[pt[:,ax]>mid]
        if not len(left) or not len(right):raise RuntimeError('non-splittable edge')
        pending.extend([right,left])
    return out


def build(p,v,pr,a,q,erosion,dilation,split):
    qs={f'q{k}':observed_quantile(p[v],k/100) for k in (20,30,40,60)}
    seed=v&(p<=qs[f'q{q}']);core=seed.copy()
    if dilation:core=ndimage.binary_dilation(core,structure=np.ones((dilation,dilation)))
    if erosion:core=ndimage.binary_erosion(core,structure=np.ones((erosion,erosion)))
    core&=v
    edges,labs,sizes,objects=components(core,v,a)
    sampling=adaptive_geometry(v,abs(pr['transform'].a),a)
    allocation=adaptive_geometry(v,abs(pr['transform'].a),a,
        scene_fraction_key='allocation_edge_buffer_scene_fraction',maximum_short_side_fraction_key='allocation_edge_buffer_maximum_short_side_fraction',transition_fraction_key='allocation_transition_buffer_fraction')
    r=float(sampling['support_radius_pixels']);assert r==allocation['support_radius_pixels']
    retained=np.zeros(v.shape,bool);sources=[];edge_rows=[]
    limit=a['maximum_sampling_bbox_pixels'];floor=sampling['minimum_retained_edge_pixels']
    for eid,sl in enumerate(objects,1):
        if sl is None:continue
        b=(sl[0].start,sl[0].stop,sl[1].start,sl[1].stop)
        row=dict(edge_id=eid,edge_pixels=int(sizes[eid]),row0=b[0],row1=b[1],col0=b[2],col1=b[3],buffer_bbox_pixels=0,status='small_edge',segments=0)
        if sizes[eid]<floor:edge_rows.append(row);continue
        tile=labs[sl]==eid;rr,cc=np.where(tile);points=np.column_stack((rr+b[0],cc+b[2]))
        full_bounds,dd=local_distance_crop(tile,(b[0],b[2]),r,v.shape)
        ss=(dd<=r)&v[full_bounds[0]:full_bounds[1],full_bounds[2]:full_bounds[3]]
        box=crop_bbox(ss,(full_bounds[0],full_bounds[2]));area=(box[1]-box[0])*(box[3]-box[2]);row['buffer_bbox_pixels']=area
        retained[points[:,0],points[:,1]]=True
        if area>limit and not split:row['status']='bbox_too_large';edge_rows.append(row);continue
        segments=split_points(points,r,limit) if area>limit else [points]
        row['status']='partitioned' if len(segments)>1 else 'candidate';row['segments']=len(segments)
        for si,pts in enumerate(segments):
            if len(pts)<a['minimum_edge_connected_floor_pixels']:continue
            sr0,sc0=pts.min(axis=0);sr1,sc1=pts.max(axis=0)+1
            t=np.zeros((sr1-sr0,sc1-sc0),bool);t[pts[:,0]-sr0,pts[:,1]-sc0]=True
            bounds,dist=local_distance_crop(t,(sr0,sc0),r,v.shape)
            win=(slice(bounds[0],bounds[1]),slice(bounds[2],bounds[3]));support=(dist<=r)&v[win]
            box=crop_bbox(support,(bounds[0],bounds[2]));assert (box[1]-box[0])*(box[3]-box[2])<=limit
            local=np.zeros(v.shape,bool);local[win]=core[win]&support
            if not local.any():continue
            primary=rectangle_profiles(p,v,local,box,a)
            edge_mask=np.zeros(v.shape,bool);edge_mask[pts[:,0],pts[:,1]]=True
            fallback=normal_profiles(p,v,core,edge_mask,a)
            bw=(slice(box[0],box[1]),slice(box[2],box[3]))
            weight=source_weight(window_from_crop(dist,bounds,box),allocation)*v[bw]
            sources.append(dict(candidate_id=len(sources)+1,edge_id=eid,segment_index=si,anchor_label=eid,
                anchored=False,anchor_pixels=0,box=box,support_box=box,weight=weight.astype(np.float32),
                sampling_pixels=int(support.sum()),profiles=primary,fallback_profiles=fallback,
                center_rc=pts.mean(axis=0).tolist(),edge_points=pts))
        edge_rows.append(row)
    return dict(seed=seed,core=core,edges=retained,quantiles=qs,sources=sources,sampling=sampling,allocation=allocation,
                geometry_notes=dict(q=q,erosion=erosion,dilation=dilation,oversize_split=split,minimum_parent_edge_pixels=floor),edge_audit=edge_rows)


def strip_width(short_side, fraction=(1 / 16)):
    """Nearest odd sample count to fraction*short_side, with a 3-pixel floor.

    Even-integer ties round upward. Counts specify unique transverse pixel
    centres, so the outermost-centre distance is count-1, not count.
    """
    return max(3, 2 * int(np.floor(float(short_side) * fraction / 2)) + 1)


def relative_resample(geometry, power, valid, algorithm):
    out = dict(geometry, sources=[])
    width_rows = []
    for source in geometry['sources']:
        s = dict(source)
        r0, r1, c0, c1 = s['box']
        short = min(r1-r0, c1-c0)
        width = strip_width(short)
        pts = s['edge_points']
        sr0, sc0 = pts.min(axis=0)
        sr1, sc1 = pts.max(axis=0) + 1
        edge = np.zeros((sr1-sr0, sc1-sc0), bool)
        edge[pts[:, 0]-sr0, pts[:, 1]-sc0] = True
        bounds, dist = local_distance_crop(edge, (sr0, sc0), geometry['sampling']['support_radius_pixels'], valid.shape)
        win = (slice(bounds[0], bounds[1]), slice(bounds[2], bounds[3]))
        local_core = np.zeros(valid.shape, bool)
        local_core[win] = geometry['core'][win] & (dist <= geometry['sampling']['support_radius_pixels']) & valid[win]
        profiles = []
        for raw in source['profiles']:
            record = dict(raw)
            # Exactly the frozen longitudinal stations, rounded as in cross15.
            centers = np.rint(np.asarray(raw['centers_rc']))
            offsets = np.arange(-(width//2), width//2 + 1)
            tangent = np.array([0., 1.]) if raw['axis'] == 'vertical_short_axis' else np.array([1., 0.])
            rows = np.rint(centers[:, 0, None] + offsets*tangent[0]).astype(int)
            cols = np.rint(centers[:, 1, None] + offsets*tangent[1]).astype(int)
            curve, fraction, samples = sample_strip(power, valid, local_core, rows, cols)
            record.update(profile=curve, core=fraction >= algorithm['profile_core_presence_fraction'],
                          samples=samples, centers_rc=centers, sample_rows=rows, sample_cols=cols,
                          cross_width_pixels=width, cross_width_fraction=(1 / 16))
            profiles.append(record)
        s['profiles'] = profiles
        # Original three-pixel normal fallback is unchanged, including stations.
        s['fallback_profiles'] = source['fallback_profiles']
        out['sources'].append(s)
        width_rows.append(dict(candidate_id=s['candidate_id'], edge_id=s['edge_id'],
                               segment_index=s['segment_index'], row0=r0, row1=r1, col0=c0, col1=c1,
                               short_side_pixels=short, cross_width_pixels=width, cross_fraction=(1 / 16)))
        del local_core
    return out, width_rows


def box_area(box: tuple[int, int, int, int]) -> int:
    return (box[1] - box[0]) * (box[3] - box[2])


def box_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (outer[0] <= inner[0] and outer[1] >= inner[1]
            and outer[2] <= inner[2] and outer[3] >= inner[3])


def greedy_nested_prune(boxes: dict[int, tuple[int, int, int, int]],
                        ratio_limit: float | None) -> tuple[set[int], list[dict[str, Any]]]:
    """Legacy greedy keep-largest pass.  ratio_limit=None -> drop any fully nested box."""
    removed: set[int] = []
    removed_set: set[int] = set()
    info: list[dict[str, Any]] = []
    retained: list[int] = []
    for cid in sorted(boxes, key=lambda i: box_area(boxes[i]), reverse=True):
        parent = None
        for rid in retained:
            if box_contains(boxes[rid], boxes[cid]):
                ratio = box_area(boxes[cid]) / max(box_area(boxes[rid]), 1)
                if ratio_limit is None or ratio <= ratio_limit:
                    parent = (rid, ratio)
                    break
        if parent is not None:
            removed_set.add(cid)
            info.append({"child": cid, "parent": parent[0], "ratio": parent[1],
                         "child_area": box_area(boxes[cid]), "parent_area": box_area(boxes[parent[0]])})
        else:
            retained.append(cid)
    removed = removed_set
    return removed, info


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    half = cumulative[-1] / 2
    position = min(int(np.searchsorted(cumulative, half)), len(order) - 1)
    return float(values[order][position])
