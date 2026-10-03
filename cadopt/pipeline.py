from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

from .features import extract_feature_lines
from .field_orientation import solve_orientation_field
from .field_position import detect_singularities
from .io_step import load_step_shape
from .metrics import residual_stats
from .quad_extract_project import (
    _remesh_uniform_triangles,
    compute_mesh_quality,
    compute_mesh_quality_extended,
    extract_and_project_quad_mesh,
)
from .tessellation import TessellationSettings, tessellate_shape_with_face_mapping
from .types import AppState, RemeshSettings


def _transfer_face_ids(
    old_V: np.ndarray, old_F: np.ndarray, old_face_ids: np.ndarray, new_V: np.ndarray, new_F: np.ndarray
) -> np.ndarray:
    """Fast transfer of face IDs from old mesh to new mesh using centroids + cKDTree."""
    # Centroids of original triangles
    old_centroids = np.mean(old_V[old_F], axis=1)
    tree = cKDTree(old_centroids)

    # Centroids of new triangles
    new_centroids = np.mean(new_V[new_F], axis=1)
    
    # Query nearest original triangle for each new triangle
    _dist, idx = tree.query(new_centroids)
    return old_face_ids[idx]


def _phase_enabled(k: int, enabled_phases: Iterable[int]) -> bool:
    s = set(int(x) for x in enabled_phases)
    return k in s


def run_pipeline_headless(
    step_path: Path,
    enabled_phases: Iterable[int] = (1, 2, 3, 4, 5),
    tess_linear_deflection: float = 0.25,
    tess_angular_deflection: float = 0.3,
    phase5_mode: str = "tri",
    phase5_keep_largest_component: bool = True,
    phase5_topo_target_factor: float = 1.0,
    phase5_topo_remesh_iters: int = 2,
    phase5_shade_sharp_deg: float = 45.0,
    phase5_flip_normals: bool = False,
    phase5_smooth_weld_enable: bool = True,
    phase5_smooth_weld_tol: float = 1e-6,
    phase5_smooth_weld_angle_deg: float = 35.0,
    phase5_normal_relax_iters: int = 0,
    phase5_normal_relax_strength: float = 0.2,
    phase5_normal_relax_global: bool = False,
    phase5_smooth_weld_global: bool = False,
    phase5_edge_snap: bool = True,
    phase5_face_projection: bool = True,
    remesh_settings: Optional[RemeshSettings] = None,
) -> Dict:
    """Run phases in CLI/headless mode and return state + metrics."""
    state = AppState(step_path=str(step_path))

    metrics: Dict = {
        "step_path": str(step_path),
        "phases": {
            "phase1": {"ok": False, "time_s": 0.0, "notes": ""},
            "phase2": {"ok": False, "time_s": 0.0, "notes": ""},
            "phase3": {"ok": False, "time_s": 0.0, "notes": ""},
            "phase4": {"ok": False, "time_s": 0.0, "notes": ""},
            "phase5": {"ok": False, "time_s": 0.0, "notes": ""},
        },
        "total_time_s": 0.0,
    }

    t0_total = time.perf_counter()
    shape = load_step_shape(step_path)

    # Phase 1
    if _phase_enabled(1, enabled_phases):
        t0 = time.perf_counter()
        p1 = tessellate_shape_with_face_mapping(
            shape,
            settings=TessellationSettings(
                linear_deflection=tess_linear_deflection,
                angular_deflection=tess_angular_deflection,
            ),
        )
        dt = time.perf_counter() - t0
        state.phase1 = p1
        metrics["phases"]["phase1"].update(
            {
                "ok": True,
                "time_s": dt,
                "n_verts": int(p1.V.shape[0]),
                "n_tris": int(p1.F.shape[0]),
                "notes": f"dense mesh verts={p1.V.shape[0]}, tris={p1.F.shape[0]}",
            }
        )

    # Phase 2
    if _phase_enabled(2, enabled_phases) and state.phase1 is not None:
        t0 = time.perf_counter()
        p2 = extract_feature_lines(state.phase1)
        dt = time.perf_counter() - t0
        state.phase2 = p2
        metrics["phases"]["phase2"].update(
            {
                "ok": True,
                "time_s": dt,
                "n_features": int(len(p2.polylines)),
                "n_constraints": int(p2.constrained_face_indices.shape[0]),
                "notes": f"features={len(p2.polylines)}, constraints={p2.constrained_face_indices.shape[0]}",
            }
        )

    # Phase 2.5: Proxy Regularization
    if remesh_settings is not None and state.phase1 is not None:
        t0 = time.perf_counter()
        old_V, old_F = state.phase1.V.copy(), state.phase1.F.copy()
        old_face_ids = state.phase1.tri_to_face_idx.copy()

        # TRACE: Starting Phase 2.5
        with open("pipeline.trace", "a") as f: f.write("Phase 2.5: start\n")

        # Extract feature vertices to lock them during remeshing
        fixed_v_idx = None
        if state.phase2 is not None and state.phase2.polylines:
            with open("pipeline.trace", "a") as f: f.write("Phase 2.5: extracting features\n")
            try:
                # Find vertices in V that are near the feature polyline points
                all_pts = np.vstack([p.points for p in state.phase2.polylines])
                from scipy.spatial import cKDTree
                v_tree = cKDTree(state.phase1.V)
                _, feat_indices = v_tree.query(all_pts, k=1)
                fixed_v_idx = np.unique(feat_indices).astype(np.int32)
                with open("pipeline.trace", "a") as f: f.write(f"Phase 2.5: found {len(fixed_v_idx)} fixed verts\n")
            except Exception as e:
                with open("pipeline.trace", "a") as f: f.write(f"Phase 2.5: feature extraction error: {e}\n")

        # Call metadata-synchronized remesher
        state.phase1.V, state.phase1.F, state.phase1.tri_to_face_idx = _remesh_uniform_triangles(
            state.phase1.V,
            state.phase1.F,
            remesh_settings.target_edge_length,
            state.phase1.tri_to_face_idx,
            remesh_settings.max_iters,
            fixed_v_idx=fixed_v_idx,
        )

        dt = time.perf_counter() - t0
        state.log(f"Phase 2.5: Proxy regularized results in {state.phase1.V.shape[0]} verts, {state.phase1.F.shape[0]} tris in {dt:.3f}s")

    # Phase 3
    if _phase_enabled(3, enabled_phases) and state.phase1 is not None and state.phase2 is not None:
        t0 = time.perf_counter()
        p3 = solve_orientation_field(state.phase1, state.phase2)
        dt = time.perf_counter() - t0
        state.phase3 = p3
        metrics["phases"]["phase3"].update(
            {
                "ok": True,
                "time_s": dt,
                "n_face_vectors": int(p3.face_vectors.shape[0]),
                "notes": f"orientation vectors={p3.face_vectors.shape[0]}",
            }
        )

    # Phase 4
    if _phase_enabled(4, enabled_phases) and state.phase1 is not None and state.phase3 is not None:
        t0 = time.perf_counter()
        p4 = detect_singularities(state.phase1, state.phase3)
        dt = time.perf_counter() - t0
        state.phase4 = p4
        metrics["phases"]["phase4"].update(
            {
                "ok": True,
                "time_s": dt,
                "n_singularities": int(p4.singular_points.shape[0]),
                "notes": f"singularities={p4.singular_points.shape[0]}",
            }
        )

    # Phase 5
    if _phase_enabled(5, enabled_phases) and state.phase1 is not None:
        t0 = time.perf_counter()
        flines = state.phase2.polylines if state.phase2 else None
        p5 = extract_and_project_quad_mesh(
            state.phase1,
            phase5_mode=phase5_mode,
            keep_largest_component=phase5_keep_largest_component,
            topo_uniform_target_factor=phase5_topo_target_factor,
            topo_remesh_iters=phase5_topo_remesh_iters,
            feature_lines=flines,
            split_by_face=True,
            shade_sharp_deg=float(phase5_shade_sharp_deg),
            flip_normals=phase5_flip_normals,
            smooth_weld_enable=bool(phase5_smooth_weld_enable),
            smooth_weld_tol=float(phase5_smooth_weld_tol),
            smooth_weld_angle_deg=float(phase5_smooth_weld_angle_deg),
            normal_relax_iters=int(max(0, phase5_normal_relax_iters)),
            normal_relax_strength=float(max(0.0, phase5_normal_relax_strength)),
            normal_relax_global=bool(phase5_normal_relax_global),
            smooth_weld_global=bool(phase5_smooth_weld_global),
            edge_snap=bool(phase5_edge_snap),
            face_projection=bool(phase5_face_projection),
        )
        dt = time.perf_counter() - t0
        state.phase5 = p5

        err = residual_stats(p5.projection_residual_to_face)
        q_shape = int(p5.Q.shape[1]) if p5.Q.ndim == 2 and p5.Q.shape[0] > 0 else 0
        quality = compute_mesh_quality(p5.V_quad, p5.Q)
        quality_ext = compute_mesh_quality_extended(p5.V_quad, p5.Q)
        metrics["phases"]["phase5"].update(
            {
                "ok": True,
                "time_s": dt,
                "n_verts": int(p5.V_quad.shape[0]),
                "n_faces": int(p5.Q.shape[0]),
                "face_arity": q_shape,
                "n_quads": int(p5.Q.shape[0]) if q_shape == 4 else 0,
                "n_tris_out": int(p5.Q.shape[0]) if q_shape == 3 else 0,
                "proj_err_mean": err["mean"],
                "proj_err_max": err["max"],
                "proj_err_rms": err["rms"],
                "mesh_n_components": quality["n_components"],
                "mesh_boundary_edges": quality["boundary_edges"],
                "mesh_n_degenerate_tris": quality["n_degenerate_tris"],
                "mesh_largest_component_n_faces": quality["largest_component_n_faces"],
                "mesh_edge_len_mean": quality_ext["edge_len_mean"],
                "mesh_edge_len_min": quality_ext["edge_len_min"],
                "mesh_edge_len_max": quality_ext["edge_len_max"],
                "mesh_tri_aspect_mean": quality_ext["tri_aspect_mean"],
                "mesh_tri_aspect_p95": quality_ext["tri_aspect_p95"],
                "mesh_tri_aspect_max": quality_ext["tri_aspect_max"],
                "mesh_valence_mean": quality_ext["valence_mean"],
                "mesh_valence_max": quality_ext["valence_max"],
                "shading_harmonization_count": p5.shading_diagnostics.get("harmonization_count", 0),
                "shading_max_jump_deg": p5.shading_diagnostics.get("max_tangent_normal_jump_deg", 0.0),
                "shading_edge_snapping_count": p5.shading_diagnostics.get("edge_snapping_count", 0),
                "shading_antipodal_flip_count": p5.shading_diagnostics.get("antipodal_flip_count", 0),
                "shading_chamfer_edge_collapse_count": p5.shading_diagnostics.get("chamfer_edge_collapse_count", 0),
                "shading_smooth_weld_merge_count": p5.shading_diagnostics.get("smooth_weld_merge_count", 0),
                "shading_normal_relax_iters": p5.shading_diagnostics.get("normal_relax_iters", 0),
                "shading_normal_relax_strength": p5.shading_diagnostics.get("normal_relax_strength", 0.0),
                "notes": (
                    f"out verts={p5.V_quad.shape[0]}, faces={p5.Q.shape[0]} (arity={q_shape}), "
                    f"components={quality['n_components']}, boundaries={quality['boundary_edges']}, "
                    f"shading_jump={p5.shading_diagnostics.get('max_tangent_normal_jump_deg', 0.0):.2f} deg, "
                    f"snapped={p5.shading_diagnostics.get('edge_snapping_count', 0)}, "
                    f"anti_flips={p5.shading_diagnostics.get('antipodal_flip_count', 0)}, "
                    f"smooth_weld={p5.shading_diagnostics.get('smooth_weld_merge_count', 0)}, "
                    f"chamfer_collapses={p5.shading_diagnostics.get('chamfer_edge_collapse_count', 0)}, "
                    f"proj_err(mean/max/rms)={err['mean']:.3e}/{err['max']:.3e}/{err['rms']:.3e}"
                ),
            }
        )

    metrics["total_time_s"] = time.perf_counter() - t0_total
    return {"state": state, "metrics": metrics}
