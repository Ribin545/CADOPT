from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    if F.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    nn = np.maximum(nn, 1e-12)
    return n / nn


def split_vertices_by_crease(
    V: np.ndarray,
    F: np.ndarray,
    crease_angle_deg: float = 45.0,
    base_normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duplicate vertices along sharp edges for stable smooth shading.

    Returns remeshed (V2, F2, VN2) where VN2 are per-vertex normals.
    """
    if F.size == 0:
        return V.copy(), F.copy(), np.zeros_like(V)

    crease_angle_deg = float(max(0.0, min(180.0, crease_angle_deg)))
    cos_thr = float(np.cos(np.deg2rad(crease_angle_deg)))

    F = F.astype(np.int32, copy=False)
    n_faces, arity = int(F.shape[0]), int(F.shape[1])
    fn = _face_normals(V, F)

    corner_face = []
    corner_vid = []
    vid_to_corners: Dict[int, List[int]] = {}

    for fi in range(n_faces):
        for cj in range(arity):
            vid = int(F[fi, cj])
            cid = len(corner_face)
            corner_face.append(fi)
            corner_vid.append(vid)
            vid_to_corners.setdefault(vid, []).append(cid)

    corner_new_vid = np.full((len(corner_face),), -1, dtype=np.int32)
    V_new: List[np.ndarray] = []
    VN_new: List[np.ndarray] = []

    for vid, cids in vid_to_corners.items():
        # Group incident corners by angular similarity of face normals.
        groups: List[List[int]] = []
        gnorms: List[np.ndarray] = []

        for cid in cids:
            fi = corner_face[cid]
            n = fn[fi]

            placed = False
            for gi, gavg in enumerate(gnorms):
                d = float(np.dot(n, gavg))
                if d >= cos_thr:
                    groups[gi].append(cid)
                    gsum = np.zeros((3,), dtype=np.float64)
                    for cc in groups[gi]:
                        gsum += fn[corner_face[cc]]
                    gnorms[gi] = gsum / max(np.linalg.norm(gsum), 1e-12)
                    placed = True
                    break

            if not placed:
                groups.append([cid])
                gnorms.append(n.copy())

        for gi, grp in enumerate(groups):
            new_vid = len(V_new)
            V_new.append(V[vid].copy())

            # IMPORTANT:
            # Use group-aware face normal by default so split vertices can carry
            # different shading directions across crease groups.
            n = np.asarray(gnorms[gi], dtype=np.float64)

            # If projected CAD/NURBS normals are available, blend them in strongly
            # to suppress faceting/dents while preserving crease-group variation.
            # This keeps shading anchored to analytical surface normals.
            if base_normals is not None and base_normals.shape[0] == V.shape[0]:
                b = np.asarray(base_normals[vid], dtype=np.float64)
                nb = np.linalg.norm(b)
                if nb > 1e-12:
                    b /= nb
                    # Keep directional consistency before blending.
                    if float(np.dot(n, b)) < 0.0:
                        n = -n
                    # Heavily prefer CAD normal, retain some crease-group cue.
                    n = 0.25 * n + 0.75 * b

            n = n / max(np.linalg.norm(n), 1e-12)
            VN_new.append(n)

            for cid in grp:
                corner_new_vid[cid] = new_vid

    F_new = np.zeros_like(F)
    k = 0
    for fi in range(n_faces):
        for cj in range(arity):
            F_new[fi, cj] = corner_new_vid[k]
            k += 1

    return (
        np.asarray(V_new, dtype=np.float64),
        F_new.astype(np.int32),
        np.asarray(VN_new, dtype=np.float64),
    )


def precision_attribute_weld(
    V: np.ndarray, 
    VN: Optional[np.ndarray], 
    F: np.ndarray, 
    decimals: int = 5
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    High-performance mesh sealing using vectorized spatial rounding.
    Merges vertices that share the same quantized position and normal.
    
    Args:
        V: (N, 3) vertices.
        VN: (N, 3) normals.
        F: (M, 3) triangles.
        decimals: Precision for rounding (spatial and normal).
        
    Returns:
        V_unique: Sealed vertex array.
        VN_unique: Sealed normal array.
        F_new: Updated triangle indices.
    """
    # 1. Quantize attributes
    q_v = np.round(V, decimals=decimals) + 0.0
    
    if VN is not None:
        q_vn = np.round(VN, decimals=decimals) + 0.0
        # Stack to create a unique signature (x, y, z, nx, ny, nz)
        stacked = np.hstack((q_v, q_vn))
    else:
        stacked = q_v
        
    # 2. Find unique rows and mapping
    unique_stacked, inverse_indices = np.unique(stacked, axis=0, return_inverse=True)
    
    # 3. Reconstruct
    # Actually, np.unique with return_index is safer to get the first occurrence.
    _, first_indices = np.unique(inverse_indices, return_index=True)
    V_unique = V[first_indices]
    
    VN_unique = None
    if VN is not None:
        VN_unique = VN[first_indices]
        
    # Map faces to new indices
    F_new = inverse_indices[F]
    
    return V_unique, VN_unique, F_new
