from __future__ import annotations

from typing import Optional, Tuple

import igl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from .tessellation import compute_face_centroids, compute_face_normals
from .types import Phase1Data, Phase2Data, Phase3Data


def _orthonormal_basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build a stable tangent basis (u, v) for a given unit normal n."""
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, n))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n, ref)
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        u = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        nu = np.linalg.norm(u)
    u /= nu
    v = np.cross(n, u)
    v /= max(np.linalg.norm(v), 1e-12)
    return u, v


def _direction_to_pi_periodic_angle(t: np.ndarray, u: np.ndarray, v: np.ndarray) -> Optional[float]:
    """Map a 3D tangent direction to a signless angle in [0, pi)."""
    a = float(np.dot(t, u))
    b = float(np.dot(t, v))
    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return None
    angle = float(np.arctan2(b, a))
    if angle < 0.0:
        angle += np.pi
    if angle >= np.pi:
        angle -= np.pi
    return angle


def _weighted_pi_periodic_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """Circular mean for directions where theta and theta+pi are equivalent."""
    if angles.size == 0:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    if w.size != angles.size or not np.any(w > 0.0):
        w = np.ones_like(angles, dtype=np.float64)
    x = np.sum(w * np.cos(2.0 * angles))
    y = np.sum(w * np.sin(2.0 * angles))
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return float(angles[0])
    return float(0.5 * np.arctan2(y, x))


def solve_orientation_field(
    phase1: Phase1Data,
    phase2: Phase2Data,
    penalty: float = 1e4,
    analytical_constraints: Optional[Tuple[np.ndarray, ...]] = None,
    analytical_weight_scale: float = 25.0,
) -> Phase3Data:
    """Phase 3 core: constrained orientation solve over the dense proxy mesh.
    
    Updated to support 'Analytical Flow Lines' as hard directional constraints.
    """
    V = phase1.V
    F = phase1.F

    from scipy.spatial import cKDTree

    # Build mesh operators in NumPy/SciPy space from igl.
    L = igl.cotmatrix(V, F).tocsr()
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI).tocsr()

    A = (-L + 1e-8 * M).tocsr()
    n_vertices = V.shape[0]

    face_normals = compute_face_normals(V, F)
    face_centroids = compute_face_centroids(V, F)

    n_faces = F.shape[0]
    face_u = np.zeros((n_faces, 3), dtype=np.float64)
    face_v = np.zeros((n_faces, 3), dtype=np.float64)
    for fi in range(n_faces):
        u, v = _orthonormal_basis_from_normal(face_normals[fi])
        face_u[fi] = u
        face_v[fi] = v

    # Per-face angle sample accumulation. We aggregate with pi-periodic mean
    # so signless directions (t and -t) are treated identically.
    face_angle_samples: list[list[float]] = [[] for _ in range(n_faces)]
    face_weight_samples: list[list[float]] = [[] for _ in range(n_faces)]

    def _accumulate(fid: int, t: np.ndarray, weight: float) -> None:
        if not (0 <= fid < n_faces):
            return
        t_proj = t - np.dot(t, face_normals[fid]) * face_normals[fid]
        nt = np.linalg.norm(t_proj)
        if nt <= 1e-12:
            return
        t_proj /= nt
        angle = _direction_to_pi_periodic_angle(t_proj, face_u[fid], face_v[fid])
        if angle is None:
            return
        face_angle_samples[fid].append(float(angle))
        face_weight_samples[fid].append(float(max(weight, 1e-9)))
    
    # 1. Standard feature line constraints (sharp edges)
    for fid, t in zip(phase2.constrained_face_indices, phase2.constrained_vectors):
        _accumulate(int(fid), t, 1.0)

    # 2. NEW: Analytical Flow Line constraints (Injected from NURBS)
    if analytical_constraints is not None:
        if len(analytical_constraints) == 2:
            c_pts, c_dirs = analytical_constraints
            c_w = np.ones((len(c_pts),), dtype=np.float64)
        elif len(analytical_constraints) >= 3:
            c_pts, c_dirs, c_w = analytical_constraints[:3]
            c_w = np.asarray(c_w, dtype=np.float64)
            if c_w.shape[0] != len(c_pts):
                c_w = np.ones((len(c_pts),), dtype=np.float64)
        else:
            c_pts = np.empty((0, 3), dtype=np.float64)
            c_dirs = np.empty((0, 3), dtype=np.float64)
            c_w = np.empty((0,), dtype=np.float64)

        if len(c_pts) > 0:
            tree = cKDTree(face_centroids)
            _, f_indices = tree.query(c_pts)
            for fid, t, ww in zip(f_indices, c_dirs, c_w):
                _accumulate(int(fid), t, analytical_weight_scale * float(max(ww, 1e-9)))

    face_target_angle = np.zeros((n_faces,), dtype=np.float64)
    target_count = np.zeros((n_faces,), dtype=np.float64)
    for fi in range(n_faces):
        if not face_angle_samples[fi]:
            continue
        aa = np.asarray(face_angle_samples[fi], dtype=np.float64)
        ww = np.asarray(face_weight_samples[fi], dtype=np.float64)
        face_target_angle[fi] = _weighted_pi_periodic_mean(aa, ww)
        target_count[fi] = float(np.sum(ww))

    mask = target_count > 0.0

    # Lift face target angles to vertices by averaging incident constrained faces.
    b = np.zeros((n_vertices,), dtype=np.float64)
    w = np.zeros((n_vertices,), dtype=np.float64)
    for fi in np.where(mask)[0]:
        angle = face_target_angle[fi]
        for vid in F[fi]:
            b[vid] += angle * target_count[fi]
            w[vid] += target_count[fi]

    wmask = w > 0
    b[wmask] = b[wmask] / w[wmask]

    # Soft constraints: add diagonal penalty on constrained vertices.
    diag_w = (penalty * w).astype(np.float64)
    rhs = penalty * w * b

    # Remove nullspaces robustly by anchoring one vertex per connected component.
    # We use a higher anchor weight (max(1.0, penalty * 1)) to ensure stability.
    adjacency = A.copy().tocsr()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    n_comp, labels = connected_components(adjacency, directed=False, return_labels=True)

    anchor_weight = max(1.0, penalty * 1.0)
    for comp_id in range(n_comp):
        vids = np.where(labels == comp_id)[0]
        if vids.size == 0:
            continue

        # Prefer a constrained vertex in this component; otherwise first vertex.
        local_w = w[vids]
        if np.any(local_w > 0):
            anchor_vid = int(vids[np.argmax(local_w)])
            anchor_val = float(b[anchor_vid])
        else:
            anchor_vid = int(vids[0])
            anchor_val = 0.0

        diag_w[anchor_vid] += anchor_weight
        rhs[anchor_vid] += anchor_weight * anchor_val

    W = sp.diags(diag_w, offsets=0, shape=(n_vertices, n_vertices), format="csr")
    I = sp.identity(n_vertices, format="csr", dtype=np.float64)
    
    # 1e-4 * I provides extremely high-strength numerical stabilization.
    # This guarantees the matrix is strictly diagonally dominant.
    A_sys = A + W + 1e-4 * I
    
    try:
        vertex_angles = spla.spsolve(A_sys, rhs)
    except Exception as e:
        print(f"[Phase 3] Sparse solve error: {e}. Falling back to LSQR.")
        vertex_angles = np.full(n_vertices, np.nan)

    # Robust fallbacks when direct solve returns NaN/Inf or singular output.
    if (not np.isfinite(vertex_angles).all()) or np.isnan(vertex_angles).any():
        vertex_angles = spla.lsqr(A_sys, rhs, atol=1e-10, btol=1e-10, iter_lim=5000)[0]

    if (not np.isfinite(vertex_angles).all()) or np.isnan(vertex_angles).any():
        # Final deterministic fallback to guarantee downstream finite values.
        vertex_angles = np.asarray(b, dtype=np.float64)

    # Clean any residual non-finite values defensively.
    vertex_angles = np.nan_to_num(vertex_angles, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert vertex scalar field to per-face angle and 3D vector.
    face_angles = vertex_angles[F].mean(axis=1)
    face_vectors = np.cos(face_angles)[:, None] * face_u + np.sin(face_angles)[:, None] * face_v

    return Phase3Data(
        face_centroids=face_centroids,
        face_vectors=face_vectors,
        face_tangent_u=face_u,
        face_tangent_v=face_v,
        face_angles=face_angles,
        vertex_angles=vertex_angles,
    )
