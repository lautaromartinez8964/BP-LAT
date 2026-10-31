"""Mechanically extract the required functions from the research workspace.

This helper is for maintainers of the originating workspace only. The emitted
package modules are self-contained and do not need the research repository.
"""
import ast
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "src" / "bplat"


def extract(path, names):
    text = (ROOT / path).read_text(encoding="utf8")
    lines = text.splitlines(keepends=True)
    nodes = {node.name: node for node in ast.parse(text).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return "\n\n".join("".join(lines[nodes[name].lineno - 1:nodes[name].end_lineno])
                       for name in names)


header = '"""Ported from the project research implementation; see PROVENANCE.md."""\nfrom __future__ import annotations\nfrom typing import Any\nimport numpy as np\nfrom scipy import ndimage\nfrom scipy.signal import find_peaks\nfrom scipy.stats import rankdata\nfrom skimage.feature import canny\n'
parts = [header,
    extract("hermes/scripts/run_scpp2_continuous_support_blend_multiprofile_20260831.py",
            ["adaptive_geometry", "resolve_domain_algorithm", "source_weight", "principal_axis"]),
    extract("hermes/scripts/run_rectangle_multiprofile_l1_e5_20260826.py", ["contiguous_runs"]),
    extract("z_flood_robust/scpp2_native_research.py",
            ["observed_quantile", "sample_strip", "rectangle_profiles", "normal_profiles",
             "equal_error_threshold", "profile_run", "ordinal_estimate"]),
    extract("scripts/run_mozambique_single_image_candidates_20260921.py",
            ["pad_slice", "local_distance_crop", "crop_bbox", "window_from_crop"]),
    extract("scripts/experiment_poyang_seed_cross_20261009.py", ["components", "split_points", "build"]),
    extract("scripts/run_spcat_relative_cross_20261010.py", ["strip_width", "relative_resample"]),
    extract("scripts/experiment_nested_bbox_prune_20260913.py", ["box_area", "box_contains", "greedy_nested_prune"]),
    extract("scripts/experiment_m1_e5_variants_20260913.py", ["weighted_median"]),
]
output = "\n\n".join(parts)
for before, after in {
    "historical.principal_axis": "principal_axis", "old_profile.contiguous_runs": "contiguous_runs",
    "method.observed_quantile": "observed_quantile", "method.rectangle_profiles": "rectangle_profiles",
    "method.normal_profiles": "normal_profiles", "method.sample_strip": "sample_strip",
    "m1.historical.adaptive_geometry": "adaptive_geometry", "m1.historical.source_weight": "source_weight",
    "gb.local_distance_crop": "local_distance_crop", "gb.crop_bbox": "crop_bbox",
    "gb.window_from_crop": "window_from_crop", "base.gb.local_distance_crop": "local_distance_crop",
    "base.m1.historical.source_weight": "source_weight", "base.gb.window_from_crop": "window_from_crop",
    "base.method.sample_strip": "sample_strip", "base.local_distance_crop": "local_distance_crop",
    "base.sample_strip": "sample_strip", "FRACTION": "(1 / 16)",
}.items():
    output = output.replace(before, after)
(OUT / "research_geometry.py").write_text(output, encoding="utf8", newline="\n")
raw = yaml.safe_load((ROOT / "config/nazira_scpp2_object_anchor_edge_normal_300m_r3_20260831.yaml").read_text(encoding="utf8"))["algorithm"]
native = json.loads((ROOT / "config/scpp2_native_profile_research_20260907.json").read_text(encoding="utf8"))["native_contract"]
for key in ("profile_cross_statistic", "global_backbone_aggregation", "profile_bank_minimum_relative_contrast", "local_profile_minimum_passed", "profile_threshold_max_relative_mad"):
    raw[key] = native[key]
raw["edge_buffer_scene_fraction"] = raw["allocation_edge_buffer_scene_fraction"]
raw["edge_buffer_maximum_short_side_fraction"] = raw["allocation_edge_buffer_maximum_short_side_fraction"]
raw["allocation_anchor_full_e5_component"] = False
(OUT / "research_config.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf8")
