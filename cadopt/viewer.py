from __future__ import annotations

from typing import List

import numpy as np
import polyscope as ps

from .types import AppState, FeaturePolyline, Phase1Data, Phase2Data, Phase3Data, Phase4Data, Phase5Data


def init_viewer() -> None:
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")


def show_phase1_mesh(state: AppState, phase1: Phase1Data) -> None:
    if "dense_mesh" in state.handles:
        state.handles["dense_mesh"].remove()
    dense = ps.register_surface_mesh("Dense Proxy Mesh", phase1.V, phase1.F)
    dense.set_smooth_shade(True)
    state.handles["dense_mesh"] = dense


def _polyline_edges(n: int) -> np.ndarray:
    if n < 2:
        return np.zeros((0, 2), dtype=np.int32)
    e = np.zeros((n - 1, 2), dtype=np.int32)
    e[:, 0] = np.arange(0, n - 1)
    e[:, 1] = np.arange(1, n)
    return e


def show_phase2_features(state: AppState, phase2: Phase2Data) -> None:
    # 1. Purge all old feature handles
    old_names = [k for k in state.handles.keys() if k.startswith("features_")]
    for name in old_names:
        try:
            state.handles[name].remove()
        except: pass
        del state.handles[name]

    # 2. Group polylines by kind
    from collections import defaultdict
    groups = defaultdict(list)
    for poly in phase2.polylines:
        groups[poly.edge_kind].append(poly.points)

    # 3. Register one consolidated network per kind
    for kind, point_lists in groups.items():
        if not point_lists: continue
        
        all_pts = []
        all_edges = []
        curr_idx = 0
        for pts in point_lists:
            n = pts.shape[0]
            if n < 2: continue
            all_pts.append(pts)
            # Create edges for this segment
            seg_edges = np.zeros((n - 1, 2), dtype=np.int32)
            seg_edges[:, 0] = np.arange(curr_idx, curr_idx + n - 1)
            seg_edges[:, 1] = np.arange(curr_idx + 1, curr_idx + n)
            all_edges.append(seg_edges)
            curr_idx += n
            
        if not all_pts: continue
        
        v_final = np.vstack(all_pts)
        e_final = np.vstack(all_edges)
        name = f"features_{kind}"
        
        cn = ps.register_curve_network(name, v_final, e_final)
        cn.set_radius(0.0005, relative=True)
        
        if kind == "boundary":
            cn.set_color((1.0, 0.6, 0.2))  # Orange
        elif kind == "flow":
            cn.set_color((0.1, 0.8, 0.9))  # Cyan (Analytical)
            cn.set_radius(0.0003, relative=True)
        else:
            cn.set_color((0.9, 0.2, 0.2))  # Red (Sharp)
            
        state.handles[name] = cn


def show_phase3_orientation(state: AppState, phase3: Phase3Data) -> None:
    dense = state.handles.get("dense_mesh", None)
    if dense is None:
        return
    try:
        if "orientation_vectors" in state.handles:
            state.handles["orientation_vectors"].remove()
    except Exception:
        pass
    q = dense.add_vector_quantity(
        "Orientation Field",
        phase3.face_vectors,
        defined_on="faces",
        enabled=True,
    )
    q.set_radius(0.0005, relative=True)
    q.set_length_scale(0.04, relative=True)
    state.handles["orientation_vectors"] = q


def show_phase4_singularities(state: AppState, phase4: Phase4Data) -> None:
    if "singularity_cloud" in state.handles:
        state.handles["singularity_cloud"].remove()

    if phase4.singular_points.shape[0] == 0:
        state.log("[Phase 4] No singularities detected by current heuristic.")
        return

    pc = ps.register_point_cloud("Singularities", phase4.singular_points)
    colors = np.zeros((phase4.singular_points.shape[0], 3), dtype=np.float64)
    neg = phase4.singular_types < 0
    pos = ~neg
    colors[neg] = np.array([1.0, 0.1, 0.1])[None, :]   # 3-poles -> red
    colors[pos] = np.array([0.1, 0.2, 1.0])[None, :]   # 5-poles -> blue
    pc.add_color_quantity("pole_type", colors, enabled=True)
    pc.set_radius(0.006, relative=True)
    state.handles["singularity_cloud"] = pc


def show_phase5_quad_mesh(state: AppState, phase5: Phase5Data) -> None:
    if "quad_mesh" in state.handles:
        state.handles["quad_mesh"].remove()

    if "dense_mesh" in state.handles:
        state.handles["dense_mesh"].set_enabled(False)

    # polyscope supports polygon meshes for quads.
    qm = ps.register_surface_mesh("Projected Quad Mesh", phase5.V_quad, phase5.Q)
    # Use smooth shading so curved CAD regions render less faceted.
    qm.set_smooth_shade(True)
    qm.add_vector_quantity("Projected Normals", phase5.projected_normals, enabled=False)
    state.handles["quad_mesh"] = qm
