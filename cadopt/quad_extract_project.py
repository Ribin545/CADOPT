from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union

import igl
import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve, GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from scipy.spatial import cKDTree

from .types import FeaturePolyline, Phase1Data, Phase5Data


def _pair_triangles_into_quads(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy triangle pairing via shared edges to form 4-vertex quads.

    Returns:
        Q: (k,4) quad indices
        used: (n_faces,) boolean mask of triangles consumed by pairing
    """
    TT, _ = igl.triangle_triangle_adjacency(F)
    n_faces = F.shape[0]

    used = np.zeros((n_faces,), dtype=bool)
    quads: List[List[int]] = []

    for fi in range(n_faces):
        if used[fi]:
            continue
        best = -1
        for ei in range(3):
            nbr = int(TT[fi, ei])
            if nbr >= 0 and not used[nbr]:
                best = nbr
                break
        if best < 0:
            continue

        a = set(int(x) for x in F[fi])
        b = set(int(x) for x in F[best])
        shared = a.intersection(b)
        if len(shared) != 2:
            continue

        uniq = list((a.union(b)))
        if len(uniq) != 4:
            continue

        # Order approximately around centroid in local plane.
        pts = np.asarray(uniq, dtype=np.int32)
        used[fi] = True
        used[best] = True
        quads.append(pts.tolist())

    if not quads:
        return np.zeros((0, 4), dtype=np.int32), used
    return np.asarray(quads, dtype=np.int32), used


def _tri_edge_counts(F_tri: np.ndarray) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = {}
    for f in F_tri:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            counts[e] = counts.get(e, 0) + 1
    return counts


def _face_components(F_tri: np.ndarray) -> Tuple[np.ndarray, int]:
    n = int(F_tri.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int32), 0

    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for fi, f in enumerate(F_tri):
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edge_to_faces[e].append(fi)

    adj: List[List[int]] = [[] for _ in range(n)]
    for flist in edge_to_faces.values():
        if len(flist) < 2:
            continue
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                a, b = flist[i], flist[j]
                adj[a].append(b)
                adj[b].append(a)

    comp = np.full((n,), -1, dtype=np.int32)
    cid = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        q = deque([s])
        comp[s] = cid
        while q:
            u = q.popleft()
            for v in adj[u]:
                if comp[v] < 0:
                    comp[v] = cid
                    q.append(v)
        cid += 1

    return comp, cid


def _degenerate_tri_count(V: np.ndarray, F_tri: np.ndarray, eps: float = 1e-18) -> int:
    if F_tri.size == 0:
        return 0
    p0 = V[F_tri[:, 0]]
    p1 = V[F_tri[:, 1]]
    p2 = V[F_tri[:, 2]]
    area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    return int(np.count_nonzero(area2 <= eps))


def _orient_triangles_consistently(F_tri: np.ndarray) -> np.ndarray:
    """Orient triangle winding consistently per connected component.

    For each pair of adjacent faces sharing an edge, enforce opposite direction
    traversal along the shared edge.
    """
    if F_tri.size == 0:
        return F_tri.astype(np.int32, copy=True)

    F = F_tri.astype(np.int32, copy=True)
    n = F.shape[0]

    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for fi, f in enumerate(F):
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edge_to_faces[e].append(fi)

    def _edge_dir(face: np.ndarray, u: int, v: int) -> int:
        # +1 if edge appears as u->v in this face cycle, -1 if v->u, 0 otherwise
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        cyc = ((a, b), (b, c), (c, a))
        for x, y in cyc:
            if x == u and y == v:
                return +1
            if x == v and y == u:
                return -1
        return 0

    visited = np.zeros((n,), dtype=bool)
    for seed in range(n):
        if visited[seed]:
            continue
        q = deque([seed])
        visited[seed] = True

        while q:
            ufi = q.popleft()
            fu = F[ufi]
            for ea, eb in ((int(fu[0]), int(fu[1])), (int(fu[1]), int(fu[2])), (int(fu[2]), int(fu[0]))):
                e = (ea, eb) if ea < eb else (eb, ea)
                nbrs = edge_to_faces.get(e, [])
                if len(nbrs) < 2:
                    continue

                du = _edge_dir(F[ufi], e[0], e[1])
                if du == 0:
                    continue

                for vfi in nbrs:
                    if vfi == ufi:
                        continue
                    dv = _edge_dir(F[vfi], e[0], e[1])
                    if dv == 0:
                        continue

                    # For consistent orientation across manifold edge: du == -dv
                    if du == dv:
                        F[vfi] = F[vfi][[0, 2, 1]]

                    if not visited[vfi]:
                        visited[vfi] = True
                        q.append(vfi)

    return F


def _compute_vertex_normals_from_faces(V: np.ndarray, F_faces: np.ndarray) -> np.ndarray:
    """Compute angle-weighted vertex normals (Vectorized).
    
    Significantly faster than the loop-based version for large CAD meshes.
    """
    F_tri = _triangulate_faces(F_faces)
    if F_tri.size == 0:
        return np.zeros_like(V)

    # p0, p1, p2: (n_faces, 3)
    p0 = V[F_tri[:, 0]]
    p1 = V[F_tri[:, 1]]
    p2 = V[F_tri[:, 2]]

    # Edge vectors: (n_faces, 3)
    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p0 - p2

    # Corner angles using dot products of normalized edges
    def _normalize(v):
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norm, 1e-12)

    e0n = _normalize(e0)
    e1n = _normalize(e1)
    e2n = _normalize(e2)

    # alpha (at p0), beta (at p1), gamma (at p2)
    alpha = np.arccos(np.clip(np.sum(e0n * -e2n, axis=1), -1.0, 1.0))
    beta = np.arccos(np.clip(np.sum(-e0n * e1n, axis=1), -1.0, 1.0))
    gamma = np.arccos(np.clip(np.sum(-e1n * e2n, axis=1), -1.0, 1.0))

    # Face normals
    fn = np.cross(e0, -e2)
    fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(fn_norm, 1e-12)

    # Weight face normals by angles and accumulate
    vn = np.zeros_like(V)
    np.add.at(vn, F_tri[:, 0], fn * alpha[:, np.newaxis])
    np.add.at(vn, F_tri[:, 1], fn * beta[:, np.newaxis])
    np.add.at(vn, F_tri[:, 2], fn * gamma[:, np.newaxis])

    # Normalize final vertex normals
    vn_norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(vn_norm, 1e-12)


def _triangulate_faces(F_faces: np.ndarray) -> np.ndarray:
    if F_faces.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    if F_faces.shape[1] == 3:
        return F_faces.astype(np.int32, copy=False)
    if F_faces.shape[1] == 4:
        q = F_faces.astype(np.int32, copy=False)
        t1 = q[:, [0, 1, 2]]
        t2 = q[:, [0, 2, 3]]
        return np.vstack([t1, t2]).astype(np.int32, copy=False)
    raise ValueError("Only triangle or quad faces are supported.")


def _mesh_quality(V: np.ndarray, F_faces: np.ndarray) -> Dict[str, int]:
    F_tri = _triangulate_faces(F_faces)
    comp, n_comp = _face_components(F_tri)
    edge_counts = _tri_edge_counts(F_tri)
    boundary_edges = int(sum(1 for c in edge_counts.values() if c == 1))
    deg = _degenerate_tri_count(V, F_tri)
    return {
        "n_components": int(n_comp),
        "boundary_edges": boundary_edges,
        "n_degenerate_tris": int(deg),
        "n_tri_faces_eval": int(F_tri.shape[0]),
        "n_faces_output": int(F_faces.shape[0]),
        "n_vertices_output": int(V.shape[0]),
        "largest_component_n_faces": int(np.max(np.bincount(comp)) if comp.size else 0),
    }


def _mesh_quality_float(V: np.ndarray, F_faces: np.ndarray) -> Dict[str, float]:
    """Extended topology quality metrics with aspect/length stats."""
    F_tri = _triangulate_faces(F_faces)
    if F_tri.size == 0:
        return {
            "edge_len_mean": 0.0,
            "edge_len_min": 0.0,
            "edge_len_max": 0.0,
            "tri_aspect_mean": 0.0,
            "tri_aspect_p95": 0.0,
            "tri_aspect_max": 0.0,
            "valence_mean": 0.0,
            "valence_max": 0.0,
        }

    edge_counts = _tri_edge_counts(F_tri)
    edge_lens = []
    for (a, b) in edge_counts.keys():
        la = V[int(a)]
        lb = V[int(b)]
        edge_lens.append(float(np.linalg.norm(lb - la)))
    e = np.asarray(edge_lens, dtype=np.float64)

    p0 = V[F_tri[:, 0]]
    p1 = V[F_tri[:, 1]]
    p2 = V[F_tri[:, 2]]
    l01 = np.linalg.norm(p1 - p0, axis=1)
    l12 = np.linalg.norm(p2 - p1, axis=1)
    l20 = np.linalg.norm(p0 - p2, axis=1)
    lmax = np.maximum(np.maximum(l01, l12), l20)
    lmin = np.minimum(np.minimum(l01, l12), l20)
    aspect = lmax / np.maximum(lmin, 1e-12)

    n_verts = int(V.shape[0])
    valence = np.zeros((n_verts,), dtype=np.int32)
    for (a, b) in edge_counts.keys():
        valence[int(a)] += 1
        valence[int(b)] += 1
    vv = valence.astype(np.float64)

    return {
        "edge_len_mean": float(np.mean(e)),
        "edge_len_min": float(np.min(e)),
        "edge_len_max": float(np.max(e)),
        "tri_aspect_mean": float(np.mean(aspect)),
        "tri_aspect_p95": float(np.percentile(aspect, 95.0)),
        "tri_aspect_max": float(np.max(aspect)),
        "valence_mean": float(np.mean(vv)),
        "valence_max": float(np.max(vv)),
    }


def _upsample_triangles_midpoint(
    V: np.ndarray, 
    F_tri: np.ndarray, 
    tri_face_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust pure-numpy 1-to-4 triangle subdivision via edge midpoints.

    Propagates face provenance IDs (tri_face_idx) from parent to all children.
    """
    Vw = V.astype(np.float64, copy=True)
    Fw = F_tri.astype(np.int32, copy=False)
    Tw = tri_face_idx.astype(np.int32, copy=False)

    edge_to_mid: Dict[Tuple[int, int], int] = {}
    new_vertices: List[np.ndarray] = []

    def _midpoint_idx(a: int, b: int) -> int:
        e = (a, b) if a < b else (b, a)
        idx = edge_to_mid.get(e, -1)
        if idx >= 0:
            return idx
        pa = Vw[e[0]]
        pb = Vw[e[1]]
        m = 0.5 * (pa + pb)
        idx = int(Vw.shape[0] + len(new_vertices))
        edge_to_mid[e] = idx
        new_vertices.append(m)
        return idx

    out_faces: List[List[int]] = []
    out_ids: List[int] = []
    
    for tid in range(Fw.shape[0]):
        f = Fw[tid]
        fid = int(Tw[tid])
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        ab = _midpoint_idx(a, b)
        bc = _midpoint_idx(b, c)
        ca = _midpoint_idx(c, a)
        
        # 4 sub-triangles
        out_faces.append([a, ab, ca])
        out_faces.append([ab, b, bc])
        out_faces.append([ca, bc, c])
        out_faces.append([ab, bc, ca])
        
        # All inherit parent ID
        out_ids.extend([fid, fid, fid, fid])

    if new_vertices:
        V2 = np.vstack([Vw, np.asarray(new_vertices, dtype=np.float64)])
    else:
        V2 = Vw
    F2 = np.asarray(out_faces, dtype=np.int32)
    T2 = np.asarray(out_ids, dtype=np.int32)
    return V2, F2, T2


def _remesh_uniform_triangles(
    V: np.ndarray,
    F_tri: np.ndarray,
    target_edge_length: float,
    tri_face_idx: np.ndarray,
    max_iters: int = 5,
    fixed_v_idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Isotropic remeshing loop to improve triangle quality for shading.
    
    Now strictly maintains tri_to_face_idx mapping to prevent projection artifacts.
    """
    if F_tri.size == 0:
        return V, F_tri, tri_face_idx
        
    # 0. Pre-clean
    try:
        # We must clean the face IDs too if igl removes vertices/faces
        Vw, _, _, Fw = igl.remove_duplicate_vertices(V, F_tri, 1e-7)
        # For simplicity, if cleaning removes triangles, we'll re-run a simple transfer
        if Fw.shape[0] != F_tri.shape[0]:
             # Find mapping from cleaned faces back to original faces
             # This is rare if input is already clean, but we handle it.
             old_cent = np.mean(V[F_tri], axis=1)
             new_cent = np.mean(Vw[Fw], axis=1)
             from scipy.spatial import cKDTree
             tree = cKDTree(old_cent)
             _, nn = tree.query(new_cent, k=1)
             Tw = tri_face_idx[nn].astype(np.int32)
        else:
             Tw = tri_face_idx.copy()

        # Filter zero-area triangles
        double_areas = igl.doublearea(Vw, Fw)
        mask = double_areas > 1e-12
        Fw = Fw[mask]
        Tw = Tw[mask]
        
        if Fw.ndim != 2 or Fw.shape[1] != 3:
            return V, F_tri, tri_face_idx
    except Exception:
        Vw, Fw, Tw = V.copy(), F_tri.copy(), tri_face_idx.copy()

    L = float(max(target_edge_length, 1e-9))
    fixed_coords = None
    if fixed_v_idx is not None and fixed_v_idx.size > 0:
        fixed_coords = Vw[fixed_v_idx % Vw.shape[0]].copy()

    for _ in range(int(max_iters)):
        try:
            # 1. Upsample if too coarse
            avg = float(igl.avg_edge_length(Vw, Fw))
            if avg > 1.4 * L and Fw.shape[0] < 200_000:
                Vw, Fw, Tw = _upsample_triangles_midpoint(Vw, Fw, Tw)

            # 2. Decimate if too dense
            avg = float(igl.avg_edge_length(Vw, Fw))
            if avg < 0.7 * L:
                total_area = float(np.sum(igl.doublearea(Vw, Fw))) * 0.5
                target_f = int(total_area / (L*L * 0.433))
                target_f = max(target_f, 5000)
                if target_f < Fw.shape[0]:
                    # J contains the birth indices (mapping to original Faces)
                    success, Vw_d, Fw_d, _, J = igl.decimate(Vw, Fw, target_f)
                    if success:
                        Vw, Fw = Vw_d, Fw_d.astype(np.int32)
                        # Update face IDs using birth mapping J
                        Tw = Tw[J].astype(np.int32)

            # 3. Simple Tangential Smoothing (Laplacian)
            Vw = igl.laplacian_smoothing(Vw, Fw, "tangential", 0.5)
            
            # 4. Snap back fixed vertices if they drifted
            if fixed_coords is not None:
                from scipy.spatial import cKDTree
                tree = cKDTree(Vw)
                _, nn_idx = tree.query(fixed_coords, k=1)
                Vw[nn_idx] = fixed_coords
        except Exception:
            break

    return Vw, Fw, Tw

def _refine_bad_aspect_triangles(
    V: np.ndarray,
    F_tri: np.ndarray,
    tri_face_idx: np.ndarray,
    aspect_threshold: float = 25.0,
    max_splits: int = 15000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conforming refinement of skinny triangles via longest-edge split set.

    Uses edge-consistent red/green triangle subdivision so adjacency stays
    conforming (no T-junction cracks in topology).
    """
    if F_tri.shape[0] == 0 or max_splits <= 0:
        return V, F_tri.astype(np.int32, copy=True), tri_face_idx.astype(np.int32, copy=True)

    Vw = V.astype(np.float64, copy=True)
    Fw = F_tri.astype(np.int32, copy=True)
    Tw = tri_face_idx.astype(np.int32, copy=True)

    def _aspect_of_face(f: np.ndarray) -> float:
        p0 = Vw[int(f[0])]
        p1 = Vw[int(f[1])]
        p2 = Vw[int(f[2])]
        l01 = float(np.linalg.norm(p1 - p0))
        l12 = float(np.linalg.norm(p2 - p1))
        l20 = float(np.linalg.norm(p0 - p2))
        lmax = max(l01, l12, l20)
        lmin = max(min(l01, l12, l20), 1e-12)
        return lmax / lmin

    split_edges: set[Tuple[int, int]] = set()
    for i in range(Fw.shape[0]):
        if len(split_edges) >= int(max_splits):
            break
        f = Fw[i]
        asp = _aspect_of_face(f)
        if asp <= float(aspect_threshold):
            continue
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        l_ab = float(np.linalg.norm(Vw[b] - Vw[a]))
        l_bc = float(np.linalg.norm(Vw[c] - Vw[b]))
        l_ca = float(np.linalg.norm(Vw[a] - Vw[c]))
        if l_ab >= l_bc and l_ab >= l_ca:
            e = (a, b)
        elif l_bc >= l_ab and l_bc >= l_ca:
            e = (b, c)
        else:
            e = (c, a)
        e = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
        split_edges.add(e)

    if not split_edges:
        return Vw, Fw, Tw

    edge_mid: Dict[Tuple[int, int], int] = {}

    # Build all required midpoints first (deterministic indices).
    new_pts: List[np.ndarray] = []
    for e in split_edges:
        edge_mid[e] = int(Vw.shape[0] + len(new_pts))
        new_pts.append(0.5 * (Vw[e[0]] + Vw[e[1]]))
    if new_pts:
        Vw = np.vstack([Vw, np.asarray(new_pts, dtype=np.float64)])

    new_faces: List[List[int]] = []
    new_tri_face_ids: List[int] = []

    for i in range(Fw.shape[0]):
        a, b, c = int(Fw[i, 0]), int(Fw[i, 1]), int(Fw[i, 2])
        fid = int(Tw[i])
        e_ab = (a, b) if a < b else (b, a)
        e_bc = (b, c) if b < c else (c, b)
        e_ca = (c, a) if c < a else (a, c)
        s_ab = e_ab in split_edges
        s_bc = e_bc in split_edges
        s_ca = e_ca in split_edges

        nsplit = int(s_ab) + int(s_bc) + int(s_ca)
        if nsplit == 0:
            new_faces.append([a, b, c])
            new_tri_face_ids.append(fid)
            continue

        m_ab = edge_mid[e_ab] if s_ab else -1
        m_bc = edge_mid[e_bc] if s_bc else -1
        m_ca = edge_mid[e_ca] if s_ca else -1

        if nsplit == 1:
            if s_ab:
                new_faces.extend([[a, m_ab, c], [m_ab, b, c]])
            elif s_bc:
                new_faces.extend([[b, m_bc, a], [m_bc, c, a]])
            else:  # s_ca
                new_faces.extend([[c, m_ca, b], [m_ca, a, b]])
            new_tri_face_ids.extend([fid, fid])
            continue

        if nsplit == 2:
            if s_ab and s_bc:
                new_faces.extend([[b, m_bc, m_ab], [a, m_ab, c], [m_ab, m_bc, c]])
            elif s_bc and s_ca:
                new_faces.extend([[c, m_ca, m_bc], [b, m_bc, a], [m_bc, m_ca, a]])
            else:  # s_ca and s_ab
                new_faces.extend([[a, m_ab, m_ca], [c, m_ca, b], [m_ca, m_ab, b]])
            new_tri_face_ids.extend([fid, fid, fid])
            continue

        # nsplit == 3
        new_faces.extend(
            [
                [a, m_ab, m_ca],
                [b, m_bc, m_ab],
                [c, m_ca, m_bc],
                [m_ab, m_bc, m_ca],
            ]
        )
        new_tri_face_ids.extend([fid, fid, fid, fid])

    F2 = np.asarray(new_faces, dtype=np.int32)
    T2 = np.asarray(new_tri_face_ids, dtype=np.int32)
    return Vw, F2, T2


def _simplify_intra_face_micro_edges(
    V: np.ndarray,
    F_tri: np.ndarray,
    tri_face_idx: np.ndarray,
    feature_lines: Optional[List[FeaturePolyline]] = None,
    max_collapses: int = 2000,
    short_edge_factor: float = 0.35,
    min_face_tri_count: int = 8,
    max_dihedral_deg: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Conservative local simplification for over-segmented smooth/chamfer regions.

    Strategy:
    - collapse only *intra-CAD-face* interior edges,
    - avoid feature/sharp edges,
    - collapse only very short edges relative to the local CAD-face median,
    - collapse only across low-dihedral pairs to preserve creases.
    """
    if F_tri.shape[0] == 0 or max_collapses <= 0:
        return (
            V.astype(np.float64, copy=True),
            F_tri.astype(np.int32, copy=True),
            tri_face_idx.astype(np.int32, copy=True),
            0,
        )

    Vw = V.astype(np.float64, copy=True)
    Fw = F_tri.astype(np.int32, copy=True)
    Tw = tri_face_idx.astype(np.int32, copy=True)

    # Reuse existing sharp-edge detector to keep hard features intact.
    sharp_edges = _find_sharp_edges(Vw, Fw, feature_lines=feature_lines, angle_threshold_deg=30.0)

    # Face normals for dihedral checks.
    p0 = Vw[Fw[:, 0]]
    p1 = Vw[Fw[:, 1]]
    p2 = Vw[Fw[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn_mag = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.divide(fn, np.maximum(fn_mag, 1e-12))
    cos_dih = float(np.cos(np.deg2rad(max_dihedral_deg)))

    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for fi, f in enumerate(Fw):
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edge_to_faces[e].append(fi)

    fid_tri_counts = Counter(int(x) for x in Tw.tolist())
    fid_edge_lens: Dict[int, List[float]] = defaultdict(list)

    # Gather edge length distribution per CAD face for local thresholding.
    for (u, v), flist in edge_to_faces.items():
        if len(flist) != 2:
            continue
        f0, f1 = int(flist[0]), int(flist[1])
        fid0, fid1 = int(Tw[f0]), int(Tw[f1])
        if fid0 != fid1:
            continue
        if fid_tri_counts[fid0] < int(min_face_tri_count):
            continue
        if (u, v) in sharp_edges:
            continue
        n0 = fn[f0]
        n1 = fn[f1]
        if float(np.dot(n0, n1)) < cos_dih:
            continue
        L = float(np.linalg.norm(Vw[v] - Vw[u]))
        fid_edge_lens[fid0].append(L)

    if not fid_edge_lens:
        return Vw, Fw, Tw, 0

    fid_med: Dict[int, float] = {}
    for fid, ls in fid_edge_lens.items():
        if ls:
            fid_med[fid] = float(np.median(np.asarray(ls, dtype=np.float64)))

    if not fid_med:
        return Vw, Fw, Tw, 0

    candidates: List[Tuple[float, int, int]] = []
    for (u, v), flist in edge_to_faces.items():
        if len(flist) != 2:
            continue
        f0, f1 = int(flist[0]), int(flist[1])
        fid0, fid1 = int(Tw[f0]), int(Tw[f1])
        if fid0 != fid1:
            continue
        if fid0 not in fid_med:
            continue
        if (u, v) in sharp_edges:
            continue
        n0 = fn[f0]
        n1 = fn[f1]
        if float(np.dot(n0, n1)) < cos_dih:
            continue
        L = float(np.linalg.norm(Vw[v] - Vw[u]))
        if L <= float(short_edge_factor) * fid_med[fid0]:
            candidates.append((L, int(u), int(v)))

    if not candidates:
        return Vw, Fw, Tw, 0

    candidates.sort(key=lambda x: x[0])

    parent = np.arange(Vw.shape[0], dtype=np.int32)

    def _find(x: int) -> int:
        y = int(x)
        while parent[y] != y:
            parent[y] = parent[parent[y]]
            y = int(parent[y])
        return y

    collapsed = 0
    for _, a, b in candidates:
        if collapsed >= int(max_collapses):
            break
        ra = _find(a)
        rb = _find(b)
        if ra == rb:
            continue
        keep = ra if ra < rb else rb
        drop = rb if keep == ra else ra
        parent[drop] = keep
        collapsed += 1

    if collapsed <= 0:
        return Vw, Fw, Tw, 0

    mapped = np.empty((Vw.shape[0],), dtype=np.int32)
    for i in range(Vw.shape[0]):
        mapped[i] = _find(i)

    F2 = mapped[Fw]
    keep_face = np.logical_and.reduce(
        [
            F2[:, 0] != F2[:, 1],
            F2[:, 1] != F2[:, 2],
            F2[:, 2] != F2[:, 0],
        ]
    )
    F2 = F2[keep_face]
    T2 = Tw[keep_face]

    if F2.shape[0] == 0:
        return Vw, Fw, Tw, 0

    used = np.unique(F2.reshape(-1))
    remap = {int(v): i for i, v in enumerate(used.tolist())}
    V2 = Vw[used].copy()
    F2 = np.vectorize(lambda x: remap[int(x)], otypes=[np.int32])(F2)
    return V2, F2.astype(np.int32), T2.astype(np.int32), int(collapsed)


def compute_mesh_quality(V: np.ndarray, F_faces: np.ndarray) -> Dict[str, int]:
    """Public wrapper for mesh quality diagnostics on triangle/quad faces."""
    return _mesh_quality(V, F_faces)


def compute_mesh_quality_extended(V: np.ndarray, F_faces: np.ndarray) -> Dict[str, float]:
    """Public wrapper with extended continuous quality metrics."""
    out: Dict[str, float] = {}
    out.update({k: float(v) for k, v in _mesh_quality(V, F_faces).items()})
    out.update(_mesh_quality_float(V, F_faces))
    return out

def _stitch_t_junctions(V: np.ndarray, F: np.ndarray, FID: np.ndarray, N: Optional[np.ndarray] = None, tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Identify vertices that lie on the edges of other triangles and split the edges.
    Resolves T-junction rips that cause shading corruption.
    """
    import igl
    from scipy.spatial import cKDTree
    
    # 1. Edge extraction
    import igl
    # Use robust numpy-based unique edge extraction
    raw_edges = np.sort(igl.edges(F.astype(np.int64)), axis=1)
    E = np.unique(raw_edges, axis=0)
    tree = cKDTree(V)
    
    edge_splits = defaultdict(list)
    
    # 2. Find all (V, Edge) pairs that form a T-junction
    for ei in range(E.shape[0]):
        v1_idx, v2_idx = E[ei]
        p1 = V[v1_idx]
        p2 = V[v2_idx]
        l_vec = p2 - p1
        l_sq = np.sum(l_vec**2)
        if l_sq < 1e-12: continue
        
        # Search radius: len/2 + tol
        center = (p1 + p2) / 2.0
        radius = np.sqrt(l_sq) / 2.0 + tol
        candidates = tree.query_ball_point(center, radius)
        
        for v_idx in candidates:
            if v_idx == v1_idx or v_idx == v2_idx: continue
            p = V[v_idx]
            
            # Parametric projection t onto p1-p2
            t = np.dot(p - p1, l_vec) / l_sq
            if 0.001 < t < 0.999: # Interior to segment
                proj = p1 + t * l_vec
                dist = np.linalg.norm(p - proj)
                if dist < tol:
                    edge_splits[tuple(sorted((v1_idx, v2_idx)))].append((t, int(v_idx)))
                    
    # 3. Rebuild faces with subdivisions
    new_V = list(V)
    new_F = []
    new_FID = []
    new_N = list(N) if N is not None else None 
    
    for i in range(F.shape[0]):
        f = F[i]
        fid = FID[i]
        
        # Sort split vertices along each edge directedly
        def get_sorted_splits(idx1, idx2):
            key = tuple(sorted((idx1, idx2)))
            raw = edge_splits.get(key, [])
            if not raw: return []
            s = sorted(raw, key=lambda x: x[0]) # Ascending t (from A to B)
            if idx1 > idx2: # Edge directed B to A, so descending t
                s = s[::-1]
            return [x[1] for x in s]
            
        boundary = [
            f[0], *get_sorted_splits(f[0], f[1]),
            f[1], *get_sorted_splits(f[1], f[2]),
            f[2], *get_sorted_splits(f[2], f[0])
        ]
        
        # Robust triangulation of the boundary loop
        if len(boundary) == 3:
            new_F.append(boundary)
            new_FID.append(fid)
        else:
            for j in range(1, len(boundary) - 1):
                new_F.append([boundary[0], boundary[j], boundary[j+1]])
                new_FID.append(fid)
                
    return (
        np.array(new_V, dtype=np.float64), 
        np.array(new_F, dtype=np.int32), 
        np.array(new_FID, dtype=np.int32),
        np.array(new_N, dtype=np.float64) if new_N is not None else None
    )


def _merge_coincident_vertices(
    V: np.ndarray, 
    F: np.ndarray, 
    FID: np.ndarray, 
    N: Optional[np.ndarray] = None,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Merge vertices within tolerance to ensure a conformal manifold.
    """
    if V.shape[0] == 0:
        return V, F, FID, N
        
    try:
        # igl.remove_duplicate_vertices works on (N, 3) trilist.
        # Signature: (SV, SVI, SVJ, SF) = remove_duplicate_vertices(V, F, eps)
        working_V = V.astype(np.float64)
        working_F = F.astype(np.int64)
        is_quad = (F.shape[1] == 4)
        
        if is_quad:
            # Triangulate for IGL vertex merging
            t1 = working_F[:, [0, 1, 2]]
            t2 = working_F[:, [0, 2, 3]]
            tri_F = np.vstack([t1, t2])
            SV, SVI, SVJ, SF = igl.remove_duplicate_vertices(working_V, tri_F, tol)
            # Reconstruct quads using the vertex map SVJ (remapping indices)
            F_final = SVJ[working_F].astype(np.int32)
            
            # Remap auxiliary data: FID and N are face-based or vertex-based?
            # Actually, FID is face-based and vertex merging does not change F.shape[0].
            # However, if we were doing normal welding, we'd need to remap N. 
            # But here N is passed as None in the early stages, or it's vertex-normals.
            # If N is vertex-normals, we MUST remap them using SVI.
            new_N = N[SVI] if N is not None else None
            return SV, F_final, FID, new_N
        else:
            SV, SVI, SVJ, SF = igl.remove_duplicate_vertices(working_V, working_F, tol)
            # SF is the updated face array for triangles. face count stays same.
            new_N = N[SVI] if N is not None else None
            return SV, SF.astype(np.int32), FID, new_N
            
    except Exception:
        # Fallback: No-op
        return V, F, FID


def _find_sharp_edges(
    V: np.ndarray,
    F: np.ndarray,
    feature_lines: Optional[List[FeaturePolyline]] = None,
    angle_threshold_deg: float = 45.0,
) -> set[Tuple[int, int]]:
    """Identify edges that should be treated as sharp/discontinuous (within-face fallback).
    
    In the new provenance-driven pipeline, this is only used for intra-face ridges.
    """
    sharp_set: set[Tuple[int, int]] = set()
    
    # 1. Add edges from feature lines (Robust Segment-Aware Detection)
    if feature_lines:
        from scipy.spatial import cKDTree
        v_tree = cKDTree(V)
        
        # Pre-identify all vertices that lie on ANY feature segment
        vert_on_feature = np.full(V.shape[0], -1, dtype=np.int32)
        
        for p_idx, poly in enumerate(feature_lines):
            pts = poly.points
            for i in range(len(pts) - 1):
                A, B = np.array(pts[i]), np.array(pts[i+1])
                # Vectorized point-to-segment distance check
                L = B - A
                L_mag_sq = np.dot(L, L)
                if L_mag_sq < 1e-12: 
                    continue
                
                # OPTIMIZATION: Only check vertices near the segment
                midpoint = (A + B) * 0.5
                radius = np.sqrt(L_mag_sq) * 0.5 + 1e-4
                candidates = v_tree.query_ball_point(midpoint, radius)
                if not candidates: 
                    continue
                
                V_cand = V[candidates]
                AP = V_cand - A
                t = np.clip(np.dot(AP, L) / L_mag_sq, 0.0, 1.0)
                proj = A + t[:, np.newaxis] * L
                dist_sq = np.sum((V_cand - proj)**2, axis=1)
                
                # RESTORED TIGHT TOLERANCE: 1e-5 mm to avoid false positives after remeshing drift
                mask = dist_sq < (1e-5)**2
                for idx_in_cand, is_on in enumerate(mask):
                    if is_on:
                        vert_on_feature[candidates[idx_in_cand]] = p_idx
        
        # An edge is sharp if both vertices lie on the SAME feature line
        # Check all edges in the mesh (robust to triangles/quads)
        for f in F.tolist():
            for i in range(len(f)):
                v1, v2 = int(f[i]), int(f[(i+1)%len(f)])
                f1, f2 = vert_on_feature[v1], vert_on_feature[v2]
                if f1 >= 0 and f1 == f2:
                    sharp_set.add(tuple(sorted((v1, v2))))
                
    # 2. Add edges with high dihedral angle (the "fallback")
    if angle_threshold_deg > 0:
        # Calculate face normals
        try:
            # igl.per_face_normals requires tri-mesh. For quads, we use a custom fallback.
            if F.shape[1] == 3:
                FN = igl.per_face_normals(V, F, np.array([0.0, 0.0, 1.0]))
            else:
                # Custom normal for quads/poly
                FN = []
                for f in F:
                    pts = V[f]
                    v1 = pts[1] - pts[0]
                    v2 = pts[2] - pts[0]
                    n = np.cross(v1, v2)
                    mag = np.linalg.norm(n)
                    FN.append(n / mag if mag > 1e-12 else np.array([0.0, 0.0, 1.0]))
                FN = np.array(FN)
        except Exception:
            # Final fallback
            FN = np.zeros((F.shape[0], 3))
            FN[:, 2] = 1.0
            
        cos_thr = np.cos(np.deg2rad(angle_threshold_deg))
        
        # Edge-to-face mapping (robust to arity)
        edge_to_faces = defaultdict(list)
        for fi, f in enumerate(F.tolist()):
            for i in range(len(f)):
                e = tuple(sorted((int(f[i]), int(f[(i+1)%len(f)]))))
                edge_to_faces[e].append(fi)
        
        for e, fids in edge_to_faces.items():
            if len(fids) == 2:
                n1 = FN[fids[0]]
                n2 = FN[fids[1]]
                dot = float(np.dot(n1, n2))
                if dot < cos_thr:
                    sharp_set.add(e)
            elif len(fids) != 1:
                # Non-manifold or boundary
                sharp_set.add(e)
                
    return sharp_set


def _cluster_faces_by_smoothness(
    vertex_id: int,
    candidate_faces: List[int],
    F: np.ndarray,
    tri_face_idx: np.ndarray,
    sharp_edges: set[Tuple[int, int]],
) -> List[List[int]]:
    """Group faces incident to a vertex into 'smooth clusters'.
    
    Two faces are in the same cluster if:
    1. They share the same CAD Face ID (tri_face_idx).
    2. They share an edge incident to vertex_id that is NOT marked as sharp.
    """
    if not candidate_faces:
        return []
    
    # Build a small adjacency graph within these faces
    adj = defaultdict(list)
    for i, fi in enumerate(candidate_faces):
        f = F[fi]
        # Find which edges of this face are incident to vertex_id
        for j in range(3):
            v1 = int(f[j])
            v2 = int(f[(j+1)%3])
            if v1 == vertex_id or v2 == vertex_id:
                e = tuple(sorted((v1, v2)))
                if e not in sharp_edges:
                    # Find the neighbor face for this edge
                    # (Note: we only care about neighbors within candidate_faces)
                    for j2, fj in enumerate(candidate_faces):
                        if i == j2: continue
                        
                        # PROVENANCE CHECK: If these triangles belong to different CAD faces, 
                        # they must NOT be in the same smooth cluster. This is the V2-logic
                        # that the user preferred.
                        if tri_face_idx[fi] != tri_face_idx[fj]:
                            continue

                        f_j = F[fj]
                        # Check shared edge
                        if any(tuple(sorted((int(f_j[k]), int(f_j[(k+1)%3])))) == e for k in range(3)):
                            adj[i].append(j2)
    
    # DFS/BFS to find components
    clusters = []
    visited = [False] * len(candidate_faces)
    for i in range(len(candidate_faces)):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                curr = stack.pop()
                comp.append(candidate_faces[curr])
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            clusters.append(comp)
            
    return clusters



def _project_point_to_face_exact(p: np.ndarray, face) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D point exactly to OCC face and return projected point + exact normal.
    
    Mathematical Ground Truth: Respects face orientation (FORWARD/REVERSED) to ensure 
    normals point 'outward' relative to the CAD solid.
    """
    surf = BRep_Tool.Surface(face)
    orient = face.Orientation()
    
    projector = GeomAPI_ProjectPointOnSurf(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), surf)

    if projector.NbPoints() > 0:
        q = projector.NearestPoint()
        q_np = np.array([q.X(), q.Y(), q.Z()], dtype=np.float64)
    else:
        # Robust fallback: UV inversion via ShapeAnalysis.
        sas = ShapeAnalysis_Surface(surf)
        uv = sas.ValueOfUV(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), 1e-13)
        q = sas.Value(uv.X(), uv.Y())
        q_np = np.array([q.X(), q.Y(), q.Z()], dtype=np.float64)

    sas = ShapeAnalysis_Surface(surf)
    uv = sas.ValueOfUV(gp_Pnt(float(q_np[0]), float(q_np[1]), float(q_np[2])), 1e-13)
    props = GeomLProp_SLProps(surf, uv.X(), uv.Y(), 1, 1e-13)

    if props.IsNormalDefined():
        n = props.Normal()
        n_np = np.array([n.X(), n.Y(), n.Z()], dtype=np.float64)
        
        # Apply CAD metadata orientation
        # Ground Truth: TopAbs_REVERSED means the face is used in the opposite sense of its surface.
        if orient == TopAbs_REVERSED:
            n_np *= -1.0
            
        n_np /= max(np.linalg.norm(n_np), 1e-12)
    else:
        n_np = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return q_np, n_np


def _project_point_to_edge_exact(p: np.ndarray, edges: List) -> Tuple[np.ndarray, float]:
    """Project 3D point exactly to the nearest of provided CAD edges.
    
    Returns:
        q_np: (3,) closest point on the closest edge.
        dist: distance from p to q_np.
    """
    best_q = p.copy()
    min_dist = float("inf")
    gp_p = gp_Pnt(float(p[0]), float(p[1]), float(p[2]))

    for edge in edges:
        curve_handle, first, last = BRep_Tool.Curve(edge)
        if not curve_handle:
            continue
            
        # PROJECT TO TRIMMED BOUNDS (First/Last) to prevent spikes
        projector = GeomAPI_ProjectPointOnCurve(gp_p, curve_handle, first, last)
        
        if projector.NbPoints() > 0:
            q = projector.NearestPoint()
            dist = projector.LowerDistance()
            if dist < min_dist:
                min_dist = dist
                best_q = np.array([q.X(), q.Y(), q.Z()], dtype=np.float64)
                
    return best_q, min_dist


def _remove_degenerate_faces(V: np.ndarray, F: np.ndarray, area_tol: float = 1e-10) -> np.ndarray:
    """Identify and remove triangles with area below threshold."""
    if F.shape[0] == 0:
        return F
        
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    keep = areas > area_tol
    keep = areas > area_tol
    return F[keep], keep


def _weld_vertices_by_normal(
    V: np.ndarray, 
    F: np.ndarray, 
    FID: np.ndarray, 
    N: Optional[np.ndarray] = None,
    spatial_tol: float = 1e-6,
    angle_tol_deg: float = 1.0,
    ignore_fid: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Weld coincident vertices and calculate an area-weighted consensus normal.
    
    This replaces 'first-encountered' normal selection with a weighted average,
    guaranteeing smooth transitions on curved chamfers.
    """
    if V.shape[0] == 0:
        return V, F, FID, N
        
    if N is None:
        # Fallback to coordinate-only welding if normals are missing
        return _merge_coincident_vertices(V, F, FID, N=None, tol=spatial_tol)

    # 1. Quantize for hashing
    V_quant = np.round(V / spatial_tol) * spatial_tol
    N_quant = np.round(N, decimals=3)
    
    # 2. Identify merge groups
    lookup: Dict[Tuple, int] = {}
    v_map = np.zeros(V.shape[0], dtype=np.int32)
    new_V_base = []
    groups = [] # List of original indices for each new vertex
    
    for i in range(V.shape[0]):
        if ignore_fid:
            key = (tuple(V_quant[i]), tuple(N_quant[i]))
        else:
            key = (tuple(V_quant[i]), tuple(N_quant[i]), int(FID[i]))

        if key not in lookup:
            new_idx = len(new_V_base)
            lookup[key] = new_idx
            new_V_base.append(V[i])
            groups.append([])
        
        v_idx = lookup[key]
        v_map[i] = v_idx
        groups[v_idx].append(i)

    # 3. Calculate Triangle Areas for weighting
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    # Map face areas back to vertices
    # vertex_total_area[new_idx] = sum of areas of all faces using any old_idx in this group
    n_new = len(new_V_base)
    weighted_normals = np.zeros((n_new, 3), dtype=np.float64)
    
    # For each face, distribute its weighted normal to its 3 vertices in the new indexing
    F_new = v_map[F]
    for i in range(len(F)):
        area = face_areas[i]
        # We use the analytical normals currently associated with the vertices
        # Since these are projected from CAD, we average the three vertex normals of the face
        avg_face_normal = (N[F[i,0]] + N[F[i,1]] + N[F[i,2]]) / 3.0
        weighted_n = avg_face_normal * area
        
        weighted_normals[F_new[i, 0]] += weighted_n
        weighted_normals[F_new[i, 1]] += weighted_n
        weighted_normals[F_new[i, 2]] += weighted_n

    # 4. Final Normalization
    new_N = []
    new_FID = []
    for i in range(n_new):
        wn = weighted_normals[i]
        mag = np.linalg.norm(wn)
        if mag > 1e-15:
            new_N.append(wn / mag)
        else:
            # Fallback to first original normal in group if area is 0
            new_N.append(N[groups[i][0]])
        new_FID.append(FID[groups[i][0]])

    V_out = np.asarray(new_V_base, dtype=np.float64)
    N_out = np.asarray(new_N, dtype=np.float64)
    FID_out = np.asarray(new_FID, dtype=np.int32)
    F_out = F_new
    
    return V_out, F_out, FID_out, N_out


def _get_analytical_normal_at_point(p: np.ndarray, face) -> np.ndarray:
    """Get exact analytical normal from an OCC face at the point nearest to p."""
    surf = BRep_Tool.Surface(face)
    projector = GeomAPI_ProjectPointOnSurf(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), surf)
    if projector.NbPoints() > 0:
        u, v = projector.LowerDistanceParameters()
        props = GeomLProp_SLProps(surf, u, v, 1, 1e-7)
        if props.IsNormalDefined():
            n_gp = props.Normal()
            return np.array([n_gp.X(), n_gp.Y(), n_gp.Z()], dtype=np.float64)
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _smooth_only_coincident_weld(
    V: np.ndarray,
    F: np.ndarray,
    tri_face_idx: np.ndarray,
    N: Optional[np.ndarray],
    occ_faces: List[Any],
    spatial_tol: float = 1e-6,
    smooth_angle_deg: float = 35.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """Merge only coincident vertices that are part of the same smooth shading region.

    Uses analytical CAD normals to decide if a seam is smooth (G1/G2).
    """
    if V.shape[0] == 0 or F.shape[0] == 0:
        return V, F, tri_face_idx, N, 0

    cos_thr = float(np.cos(np.deg2rad(float(max(0.0, min(180.0, smooth_angle_deg))))))

    # Incident-face lookup per vertex (for provenance ID mapping)
    v_to_faces: Dict[int, List[int]] = defaultdict(list)
    for fi, f in enumerate(F):
        for v in f.tolist():
            v_to_faces[int(v)].append(int(fi))

    v_to_fids: Dict[int, set[int]] = {}
    for vid, faces in v_to_faces.items():
        v_to_fids[vid] = set(int(tri_face_idx[int(fi)]) for fi in faces)

    tree = cKDTree(V)
    pairs = tree.query_pairs(float(max(spatial_tol, 1e-12)))
    if not pairs:
        return V, F, tri_face_idx, N, 0

    parent = np.arange(V.shape[0], dtype=np.int32)

    def _find(x: int) -> int:
        y = int(x)
        while parent[y] != y:
            parent[y] = parent[parent[y]]
            y = int(parent[y])
        return y

    merged = 0
    for u, v in pairs:
        u_int, v_int = int(u), int(v)
        ru = _find(u_int)
        rv = _find(v_int)
        if ru == rv:
            continue

        # Criteria 1: CAD-Face Continuity
        # Only weld if they belong to different CAD faces (seam) OR if they come from the same face.
        # Actually, we mostly care about inter-face boundaries.
        fids_u = v_to_fids.get(u_int, set())
        fids_v = v_to_fids.get(v_int, set())

        # If they don't share any adjacency information, skipping (safety)
        if not fids_u or not fids_v:
            continue

        # Criteria 2: Analytical Normal Comparison
        # For each face touching u and each face touching v, check if at least one pair is 'smooth'.
        is_smooth = False
        p_avg = (V[u_int] + V[v_int]) * 0.5
        
        # Check all cross-pair combinations of analytical normals
        for fia in fids_u:
            na = _get_analytical_normal_at_point(p_avg, occ_faces[fia])
            na /= max(np.linalg.norm(na), 1e-12)
            for fib in fids_v:
                nb = _get_analytical_normal_at_point(p_avg, occ_faces[fib])
                nb /= max(np.linalg.norm(nb), 1e-12)
                
                if float(np.dot(na, nb)) >= cos_thr:
                    is_smooth = True
                    break
            if is_smooth:
                break

        if not is_smooth:
            continue

        keep = ru if ru < rv else rv
        drop = rv if keep == ru else ru
        parent[drop] = keep
        merged += 1

    if merged <= 0:
        return V, F, tri_face_idx, N, 0

    mapped = np.empty((V.shape[0],), dtype=np.int32)
    for i in range(V.shape[0]):
        mapped[i] = _find(i)

    F2 = mapped[F]
    keep_face = np.logical_and.reduce(
        [
            F2[:, 0] != F2[:, 1],
            F2[:, 1] != F2[:, 2],
            F2[:, 2] != F2[:, 0],
        ]
    )
    F2 = F2[keep_face]
    T2 = tri_face_idx[keep_face]

    if F2.shape[0] == 0:
        return V, F, tri_face_idx, N, 0

    used = np.unique(F2.reshape(-1))
    remap = {int(v): i for i, v in enumerate(used.tolist())}
    V2 = V[used].copy()
    N2 = N[used].copy() if (N is not None and N.shape[0] == V.shape[0]) else None
    F2 = np.vectorize(lambda x: remap[int(x)], otypes=[np.int32])(F2)
    return V2, F2.astype(np.int32), T2.astype(np.int32), N2, int(merged)


def _relax_vertex_normals_edge_preserving(
    N: np.ndarray,
    F: np.ndarray,
    sharp_edges: set[Tuple[int, int]],
    iters: int = 2,
    strength: float = 0.2,
    ignore_sharp: bool = False,
) -> np.ndarray:
    """Laplacian-like smoothing in normal space, optionally blocked across sharp edges."""
    if N.size == 0 or F.size == 0 or iters <= 0 or strength <= 0.0:
        return N

    lam = float(max(0.0, min(1.0, strength)))
    n_verts = int(N.shape[0])
    adj: List[set[int]] = [set() for _ in range(n_verts)]

    for f in F.tolist():
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            if not ignore_sharp and e in sharp_edges:
                continue
            adj[u].add(v)
            adj[v].add(u)

    out = N.astype(np.float64, copy=True)
    for _ in range(int(iters)):
        nxt = out.copy()
        for i in range(n_verts):
            nbrs = adj[i]
            if not nbrs:
                continue
            avg = np.mean(out[list(nbrs)], axis=0)
            cand = (1.0 - lam) * out[i] + lam * avg
            mag = np.linalg.norm(cand)
            if mag > 1e-12:
                nxt[i] = cand / mag
        out = nxt
    return out


def _re_sync_attributes(I_kept: np.ndarray, *attrs: np.ndarray) -> List[np.ndarray]:
    """Subset multiple vertex-aligned attributes using the same index map."""
    # Ensure I_kept doesn't contain -1 for subsetting
    valid_mask = I_kept >= 0
    safe_indices = I_kept[valid_mask]
    return [a[safe_indices] for a in attrs]


def _distance_point_to_face(p: np.ndarray, face) -> float:
    """Compute nearest distance from point to an OCC face surface."""
    surf = BRep_Tool.Surface(face)
    projector = GeomAPI_ProjectPointOnSurf(gp_Pnt(float(p[0]), float(p[1]), float(p[2])), surf)
    if projector.NbPoints() > 0:
        return float(projector.LowerDistance())
    return float("inf")


def extract_and_project_quad_mesh(
    phase1: Phase1Data,
    phase5_mode: str = "tri",
    keep_largest_component: bool = True,
    topo_uniform_target_factor: float = 1.0,
    topo_remesh_iters: int = 5,
    feature_lines: Optional[List[FeaturePolyline]] = None,
    split_by_face: bool = True,
    shade_sharp_deg: float = 45.0,
    flip_normals: bool = False,
    smooth_weld_enable: bool = True,
    smooth_weld_tol: float = 0.05,
    smooth_weld_angle_deg: float = 35.0,
    normal_relax_iters: int = 5,
    normal_relax_strength: float = 0.4,
    smooth_weld_global: bool = False,
    normal_relax_global: bool = False,
    edge_snap: bool = True,
    face_projection: bool = True,
) -> Phase5Data:
    """Phase 5 core: build clean mesh and snap vertices to exact OCC faces.

    Args:
        phase5_mode: 'tri' (default, robust) or 'quad' (experimental)
        keep_largest_component: keep largest connected component before projection.
    """
    V = phase1.V
    F = phase1.F
    tri_to_face_idx = phase1.tri_to_face_idx

    F_work = F.copy()
    tri_face_work = tri_to_face_idx.copy()

    if keep_largest_component and F_work.shape[0] > 0:
        comp, n_comp = _face_components(F_work)
        if n_comp > 1:
            binc = np.bincount(comp)
            # Thresholding: keep all pieces that have enough faces to be part of the model.
            # This prevents accidental deletion of unsewn housing patches.
            valid_cids = np.where(binc > 100)[0]
            if len(valid_cids) > 0:
                keep_mask = np.isin(comp, valid_cids)
                F_work = F_work[keep_mask]
                tri_face_work = tri_face_work[keep_mask]
            else:
                # Fallback to largest if nothing is above threshold
                keep_cid = int(np.argmax(binc))
                keep_mask = comp == keep_cid
                F_work = F_work[keep_mask]
                tri_face_work = tri_face_work[keep_mask]

    if F_work.shape[0] == 0:
        return Phase5Data(
            V_quad=np.zeros((0, 3), dtype=np.float64),
            Q=np.zeros((0, 3), dtype=np.int32),
            projected_normals=np.zeros((0, 3), dtype=np.float64),
            projected_face_indices=np.zeros((0,), dtype=np.int32),
            projection_residual_to_face=np.zeros((0,), dtype=np.float64),
        )

    # Keep vertex set tightly coupled to surviving faces.
    used_vids = np.unique(F_work.reshape(-1))
    remap = {int(v): i for i, v in enumerate(used_vids.tolist())}
    V_quad = V[used_vids].copy()
    F_local = np.vectorize(lambda x: remap[int(x)], otypes=[np.int32])(F_work)

    chamfer_edge_collapse_count = 0

    # Phase 5a topology-improve (tri mode): uniformize triangles before projection.
    if str(phase5_mode).lower() == "tri" and F_local.shape[0] > 0:
        V_pre = V_quad.copy()
        F_pre = F_local.astype(np.int32, copy=True)
        tri_face_pre = tri_face_work.copy()
        pre_q = _mesh_quality(V_pre, F_pre)

        try:
            cur = float(igl.avg_edge_length(V_quad, F_local.astype(np.int32)))
        except Exception:
            cur = 0.0
        if np.isfinite(cur) and cur > 1e-12:
            target_len = float(max(cur * float(topo_uniform_target_factor), 1e-9))
            V_quad, F_local, tri_face_work = _remesh_uniform_triangles(
                V_quad,
                F_local.astype(np.int32),
                target_len,
                tri_face_work,
                max_iters=int(max(1, topo_remesh_iters)),
            )

        # Re-apply connected-component filtering after remesh, since subdivision
        # and decimation can introduce tiny disconnected fragments.
        if keep_largest_component and F_local.shape[0] > 0:
            comp2, n_comp2 = _face_components(F_local.astype(np.int32))
            if n_comp2 > 1:
                keep_cid2 = int(np.argmax(np.bincount(comp2)))
                keep_mask2 = comp2 == keep_cid2
                F_local = F_local[keep_mask2]
                tri_face_work = tri_face_work[keep_mask2]

                # Compact vertices after face filtering.
                used2 = np.unique(F_local.reshape(-1))
                remap2 = {int(v): i for i, v in enumerate(used2.tolist())}
                V_quad = V_quad[used2].copy()
                F_local = np.vectorize(lambda x: remap2[int(x)], otypes=[np.int32])(F_local)

        # Accept remesh only if it does not catastrophically damage connectivity.
        post_q = _mesh_quality(V_quad, F_local.astype(np.int32))
        bad_connectivity = (
            F_local.shape[0] > int(max(1, 1.2 * F_pre.shape[0]))
            or V_quad.shape[0] > int(max(1, 1.25 * V_pre.shape[0]))
            or
            post_q["n_components"] > max(2, pre_q["n_components"] * 2)
            or post_q["boundary_edges"] > max(pre_q["boundary_edges"] + 200, int(pre_q["boundary_edges"] * 1.5))
            or post_q["n_degenerate_tris"] > max(pre_q["n_degenerate_tris"] + 50, int(pre_q["n_degenerate_tris"] * 2 + 5))
        )
        if bad_connectivity:
            V_quad = V_pre
            F_local = F_pre
            tri_face_work = tri_face_pre

        # Chamfer/feature aid: locally refine very skinny triangles.
        # Keep only if it actually improves triangle quality without connectivity regression.
        V_ref0 = V_quad.copy()
        F_ref0 = F_local.astype(np.int32, copy=True)
        T_ref0 = tri_face_work.astype(np.int32, copy=True)
        q_ref0 = _mesh_quality(V_ref0, F_ref0)
        qf_ref0 = _mesh_quality_float(V_ref0, F_ref0)

        V_ref1, F_ref1, T_ref1 = _refine_bad_aspect_triangles(
            V_ref0,
            F_ref0,
            T_ref0,
            aspect_threshold=45.0,
            max_splits=1500,
        )
        q_ref1 = _mesh_quality(V_ref1, F_ref1)
        qf_ref1 = _mesh_quality_float(V_ref1, F_ref1)

        keep_refinement = (
            q_ref1["n_components"] <= q_ref0["n_components"]
            and q_ref1["boundary_edges"] <= q_ref0["boundary_edges"] + 50
            and q_ref1["n_degenerate_tris"] <= q_ref0["n_degenerate_tris"] + 10
            and qf_ref1["tri_aspect_p95"] <= qf_ref0["tri_aspect_p95"] * 0.995
        )
        if keep_refinement:
            V_quad, F_local, tri_face_work = V_ref1, F_ref1, T_ref1

        # Conservative chamfer/smooth simplification:
        # collapse only intra-face micro-edges and accept only with topology-safe gates.
        V_s0 = V_quad.copy()
        F_s0 = F_local.astype(np.int32, copy=True)
        T_s0 = tri_face_work.astype(np.int32, copy=True)
        q_s0 = _mesh_quality(V_s0, F_s0)
        qf_s0 = _mesh_quality_float(V_s0, F_s0)

        V_s1, F_s1, T_s1, n_collapsed = _simplify_intra_face_micro_edges(
            V_s0,
            F_s0,
            T_s0,
            feature_lines=feature_lines,
            max_collapses=600,
            short_edge_factor=0.25,
            min_face_tri_count=8,
            max_dihedral_deg=20.0,
        )
        q_s1 = _mesh_quality(V_s1, F_s1)
        qf_s1 = _mesh_quality_float(V_s1, F_s1)

        keep_simplification = (
            n_collapsed > 0
            and q_s1["n_components"] <= q_s0["n_components"]
            and q_s1["largest_component_n_faces"] >= int(0.995 * q_s0["largest_component_n_faces"])
            and q_s1["boundary_edges"] <= q_s0["boundary_edges"]
            and q_s1["n_degenerate_tris"] <= q_s0["n_degenerate_tris"]
            and qf_s1["tri_aspect_p95"] <= qf_s0["tri_aspect_p95"] * 1.01
            and F_s1.shape[0] <= int(0.99 * F_s0.shape[0])
        )
        if keep_simplification:
            V_quad, F_local, tri_face_work = V_s1, F_s1, T_s1
            chamfer_edge_collapse_count = int(n_collapsed)

    # Optional quad conversion, but never at the cost of severe coverage loss.
    face_out = F_local.astype(np.int32)
    if str(phase5_mode).lower() == "quad":
        Q_local, tri_used = _pair_triangles_into_quads(F_local)
        coverage = (2.0 * float(Q_local.shape[0]) / float(F_local.shape[0])) if F_local.shape[0] else 0.0
        if Q_local.shape[0] > 0 and coverage >= 0.98:
            face_out = Q_local.astype(np.int32)

    # Ensure consistent winding for triangle output to avoid shading artifacts.
    if face_out.shape[0] > 0 and face_out.shape[1] == 3:
        face_out = _orient_triangles_consistently(face_out)

    # -------------------------------------------------------------------------
    # Final Projection & Optional Vertex Splitting
    # -------------------------------------------------------------------------
    if split_by_face:
        # Pre-process: Conformize Topology (Stitch micro-gaps / T-junctions)
        # This converts T-junctions into shared vertices so that shading remains continuous.
        V_quad, face_out, tri_face_work, _ = _stitch_t_junctions(V_quad, face_out, tri_face_work, N=None, tol=1e-5)
        V_quad, face_out, tri_face_work, _ = _merge_coincident_vertices(V_quad, face_out, tri_face_work, tol=1e-6)
        
        # Identify sharp edges to drive the splitting logic.
        # This combines CAD feature lines with a dihedral angle fallback for maximum quality.
        sharp_edges = _find_sharp_edges(
            V_quad, 
            face_out, 
            feature_lines=feature_lines, 
            angle_threshold_deg=30.0 # Standard threshold for catching non-tangent boundaries
        )
        # ---------------------------------------------------------------------
        # Pre-construction: Analytical Edge Tree for Fast Snapping
        # ---------------------------------------------------------------------
        # For 'Absolute Sharpness', we find all edges in the original shape.
        all_edges = []
        edge_samples = []
        
        exp = TopExp_Explorer(phase1.shape, TopAbs_EDGE)
        while exp.More():
            edge = topods.Edge(exp.Current())
            all_edges.append(edge)
            
            # Discretize for KDTree lookup (Density 32 is sufficient for snapping)
            curve = BRepAdaptor_Curve(edge)
            first, last = curve.FirstParameter(), curve.LastParameter()
            for t in np.linspace(first, last, 32):
                p_gp = curve.Value(float(t))
                edge_samples.append([p_gp.X(), p_gp.Y(), p_gp.Z()])
            exp.Next()
            
        edge_lookup_tree = None
        if edge_samples:
            edge_lookup_tree = cKDTree(np.array(edge_samples))

        v_to_faces = defaultdict(list)
        snapping_count = 0
        for fi, f in enumerate(face_out):
            for v in f.tolist():
                v_to_faces[int(v)].append(fi)
        
        # Cluster faces at each vertex based on smooth connectivity (non-sharp edges).
        # v_clusters[v_orig] = [ [cluster0_fids], [cluster1_fids], ... ]
        v_clusters: Dict[int, List[List[int]]] = {}
        for vid in range(V_quad.shape[0]):
            incident = v_to_faces[vid]
            v_clusters[vid] = _cluster_faces_by_smoothness(
                vid, incident, face_out, tri_face_work, sharp_edges
            )
        
        new_verts = []
        new_faces = []
        new_normals = []
        new_face_ids = []
        new_residuals = []
        
        # (v_orig, cluster_idx) -> new_vid
        v_final_map: Dict[Tuple[int, int], int] = {}
        
        for i in range(face_out.shape[0]):
            f = face_out[i]
            fid = int(tri_face_work[i])
            face_occ = phase1.occ_faces[fid]
            
            new_f = []
            for v_orig in f.tolist():
                v_orig = int(v_orig)
                # Find which cluster this face belongs to at this vertex
                cluster_idx = -1
                for ci, cl in enumerate(v_clusters[v_orig]):
                    if i in cl:
                        cluster_idx = ci
                        break
                
                key = (v_orig, cluster_idx)
                if key not in v_final_map:
                    new_idx = len(new_verts)
                    v_final_map[key] = new_idx
                    
                    p = V_quad[v_orig]
                    if face_projection:
                        q, n = _project_point_to_face_exact(p, face_occ)
                    else:
                        q = p
                        n = np.array([0.0, 0, 1.0]) 
                    
                    # --- NEW: Optimized Boundary Snapping ---
                    was_snapped = False
                    if edge_snap and edge_lookup_tree is not None:
                        d_coarse, _ = edge_lookup_tree.query(q, k=1)
                        if d_coarse < 0.2:
                             dists, idxs = edge_lookup_tree.query(q, k=5)
                             candidate_edges = [all_edges[idx // 32] for idx in idxs]
                             q_edge, d_edge = _project_point_to_edge_exact(q, candidate_edges)
                             
                             dist_from_orig = np.linalg.norm(q_edge - q)
                             if d_edge < 0.1 and dist_from_orig < 0.2: 
                                 q = q_edge
                                 snapping_count += 1
                                 was_snapped = True

                    # Precision Shading: If snapped to edge, ensure normal is not unstable
                    # (We keep face-projection normal 'n' as base, but could average later)
                    # For now, was_snapped just ensures the geometric lock.
                    # ----------------------------------------

                    new_verts.append(q)
                    new_normals.append(n)
                    new_face_ids.append(fid)
                    new_residuals.append(_distance_point_to_face(q, face_occ))
                
                new_f.append(v_final_map[key])
            new_faces.append(new_f)
            
        V_final = np.array(new_verts, dtype=np.float64)
        F_final = np.array(new_faces, dtype=np.int32)
        N_final = np.array(new_normals, dtype=np.float64)
        FID_final = np.array(new_face_ids, dtype=np.int32)
        RES_final = np.array(new_residuals, dtype=np.float64)
    else:
        # Standard contiguous projection (averages positions/normals at boundaries)
        V_final = V_quad.copy()
        F_final = face_out.copy()
        
        vert_face_candidates: Dict[int, List[int]] = defaultdict(list)
        for tid in range(F_final.shape[0]):
            fidx = int(tri_face_work[tid])
            for vid in F_final[tid].tolist():
                vert_face_candidates[int(vid)].append(fidx)

        N_final = np.zeros_like(V_final)
        FID_final = np.full((V_final.shape[0],), -1, dtype=np.int32)
        RES_final = np.zeros((V_final.shape[0],), dtype=np.float64)
        fallback_fid = int(tri_face_work[0]) if tri_face_work.size > 0 else 0

        for i in range(V_final.shape[0]):
            p = V_final[i]
            cands = vert_face_candidates.get(i, [])
            if cands:
                hist = Counter(cands)
                modal_fid = hist.most_common(1)[0][0]
                fid = modal_fid
            else:
                fid = fallback_fid
            
            face_occ = phase1.occ_faces[fid]
            q, n = _project_point_to_face_exact(p, face_occ)
            V_final[i] = q
            N_final[i] = n
            FID_final[i] = fid
            RES_final[i] = _distance_point_to_face(q, face_occ)

    # -------------------------------------------------------------------------
    # Final Shading Stabilization: Provenance-Aware Harmonization
    # -------------------------------------------------------------------------
    # We no longer "guess" the orientation (V15 failed). We trust the CAD orientation
    # from Phase 5a and then only average normals at spatially coincident points 
    # that are already approximately tangent (G1/G2 threshold).
    harmonization_count = 0
    antipodal_flip_count = 0
    max_jump = 0.0
    max_jump_all = 0.0
    max_jump_sharp = 0.0
    seam_sign_sync_count = 0
    smooth_weld_merge_count = 0
    if V_final.shape[0] > 0:
        # 1. Build spatial adjacency for coincidence (within micro-tolerance)
        tree = cKDTree(V_final)
        pairs = tree.query_pairs(1e-5) # Tight tolerance to catch intended boundaries
        
        spatial_adj = defaultdict(list)
        for u, v in pairs:
            spatial_adj[u].append(v)
            spatial_adj[v].append(u)
            
        # 2. Group coincident vertices into clusters
        visited_spatial = [False] * V_final.shape[0]
        cos_group = float(np.cos(np.deg2rad(25.0)))
        
        # If smooth_weld_global is on, we treat EVERYTHING as smooth for the harmonization pass.
        if smooth_weld_global:
            cos_smooth = -1.0
            tangent_threshold = 180.0
        else:
            cos_smooth = float(np.cos(np.deg2rad(45.0)))
            tangent_threshold = 15.0
        
        for start_vid in range(V_final.shape[0]):
            if visited_spatial[start_vid] or start_vid not in spatial_adj:
                continue
                
            cluster = []
            stack = [start_vid]
            visited_spatial[start_vid] = True
            while stack:
                curr = stack.pop()
                cluster.append(curr)
                for neighbor in spatial_adj[curr]:
                    if not visited_spatial[neighbor]:
                        visited_spatial[neighbor] = True
                        stack.append(neighbor)
            
            # 3. Within each cluster, do seam-aware sign synchronization.
            # Build orientation-agnostic normal groups (using abs(dot)) so we only
            # flip normals within near-collinear tangent groups, never across sharp turns.
            if len(cluster) > 1:
                groups: List[List[int]] = []
                gdirs: List[np.ndarray] = []

                for vid in cluster:
                    n = N_final[vid]
                    nn = np.linalg.norm(n)
                    if nn <= 1e-12:
                        continue
                    n = n / nn

                    placed = False
                    for gi, gd in enumerate(gdirs):
                        if float(abs(np.dot(n, gd))) >= cos_group:
                            groups[gi].append(int(vid))
                            # Update group direction by sign-aligned accumulation.
                            s = 1.0 if float(np.dot(n, gd)) >= 0.0 else -1.0
                            gg = gd + s * n
                            gdirs[gi] = gg / max(np.linalg.norm(gg), 1e-12)
                            placed = True
                            break

                    if not placed:
                        groups.append([int(vid)])
                        gdirs.append(n.copy())

                # Orient signs consistently inside each smooth group.
                for gi, vids in enumerate(groups):
                    if len(vids) <= 1:
                        continue
                    ref = N_final[vids[0]].copy()
                    ref /= max(np.linalg.norm(ref), 1e-12)
                    for vid in vids[1:]:
                        n = N_final[vid]
                        nn = np.linalg.norm(n)
                        if nn <= 1e-12:
                            continue
                        n = n / nn
                        if float(np.dot(ref, n)) < 0.0:
                            N_final[vid] = -n
                            seam_sign_sync_count += 1
                            # Keep legacy diagnostic key meaningful (near-antipodal events corrected).
                            if float(np.dot(ref, n)) < -0.95:
                                antipodal_flip_count += 1

                # Compare all pairs in cluster to detect boundaries
                for i in range(len(cluster)):
                    for j in range(i + 1, len(cluster)):
                        v1, v2 = cluster[i], cluster[j]
                        n1, n2 = N_final[v1], N_final[v2]
                        
                        dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
                        angle = np.degrees(np.arccos(dot))
                        max_jump_all = max(max_jump_all, angle)
                        if float(abs(dot)) >= cos_smooth:
                            max_jump = max(max_jump, angle)
                        else:
                            max_jump_sharp = max(max_jump_sharp, angle)
                        
                        if angle < tangent_threshold: # Shading Tangency Threshold
                            avg_n = (n1 + n2) / 2.0
                            mag = np.linalg.norm(avg_n)
                            if mag > 1e-12:
                                avg_n /= mag
                                N_final[v1] = avg_n
                                N_final[v2] = avg_n
                                harmonization_count += 1

        print(f"[Shading] Harmonization Pass Complete. Corrected {harmonization_count} boundary pairs.")
        print(f"[Shading] Seam sign synchronizations: {seam_sign_sync_count} (antipodal={antipodal_flip_count}).")
        print(f"[Shading] Max smooth-boundary normal deviation: {max_jump:.2f} deg (all={max_jump_all:.2f}, sharp={max_jump_sharp:.2f}).")

    # -------------------------------------------------------------------------
    # POST-PROJECTION DENOISING: Final Topology Cleanup
    # -------------------------------------------------------------------------
        n_faces_before = F_final.shape[0]
        print(f"[Denoise] Start: F={n_faces_before}, FID={tri_face_work.shape[0]}")
        
        # 1. Remove 0-area and tiny slivers (Purge Area < 1e-7)
        F_final, keep_degen = _remove_degenerate_faces(V_final, F_final, area_tol=1e-7)
        tri_face_work = tri_face_work[keep_degen]
        print(f"[Denoise] After Step 1: F={F_final.shape[0]}, FID={tri_face_work.shape[0]}")
        degenerate_count = n_faces_before - F_final.shape[0]
        
        # 2. Topological Edge Refinement (T-Junction Fix)
        # Splits long edges to match adjacent vertex density, ensuring manifold seams.
        # tri_face_work is Triangle-based provenance.
        V_final, F_final, tri_face_work, N_final = _stitch_t_junctions(
            V_final, F_final, tri_face_work, N=N_final, tol=1e-5
        )
        print(f"[Denoise] After Step 2 (Stitch): F={F_final.shape[0]}, FID={tri_face_work.shape[0]}")

        # 3. Smooth-only weld of coincident seam duplicates (never across sharp transitions).
        n_verts_before_weld = V_final.shape[0]
        if bool(smooth_weld_enable):
            V_final, F_final, tri_face_work, N_final, smooth_weld_merge_count = _smooth_only_coincident_weld(
                V_final,
                F_final,
                tri_face_work,
                N_final,
                phase1.occ_faces,
                spatial_tol=float(max(1e-12, smooth_weld_tol)),
                smooth_angle_deg=180.0 if smooth_weld_global else float(smooth_weld_angle_deg),
            )
        print(f"[Denoise] After Step 3 (Weld): F={F_final.shape[0]}, FID={tri_face_work.shape[0]}")
        
        # 3. Winding-Aware Duplicate Facet Purge (Orientation Fix)
        # We must NOT use np.sort because it destroys CCW winding (scrambling normals).
        # Instead, we "roll" each triplet to start with the smallest index. 
        # [2, 3, 1] -> [1, 2, 3] (Preserves CCW)
        # [1, 3, 2] -> [1, 3, 2] (Preserves CW/CCW distinction)
        n_faces_before_dup = F_final.shape[0]
        
        # Vectorized circular shift to canonical form 
        argmin = F_final.argmin(axis=1)
        F_canonical = F_final[np.arange(len(F_final))[:, None], (np.arange(3) + argmin[:, None]) % 3]
        
        # Get unique canonical representations
        _, unique_indices = np.unique(F_canonical, axis=0, return_index=True)
        F_final = F_final[unique_indices]
        tri_face_work = tri_face_work[unique_indices] # SYNC HERE!
        print(f"[Denoise] After Duplicate Purge: F={F_final.shape[0]}, FID={tri_face_work.shape[0]}")
        duplicate_count = n_faces_before_dup - F_final.shape[0]
        
        # 4. Sliver Pruning (Remove high-aspect "needles" like spider-webs)
        n_faces_before_sliver = F_final.shape[0]
        F_final, keep_sliver = _remove_sliver_faces(V_final, F_final, max_aspect=500.0)
        tri_face_work = tri_face_work[keep_sliver]
        sliver_count = n_faces_before_sliver - F_final.shape[0]
        print(f"[Denoise] After Step 4 (Sliver): F={F_final.shape[0]}, FID={tri_face_work.shape[0]}")

        # 5. Automated Shading Harmonization (REBOOT: Simple Baseline)
        # Ensure triangle winding follows the analytical normal direction locally.
        f_v0, f_v1, f_v2 = V_final[F_final[:, 0]], V_final[F_final[:, 1]], V_final[F_final[:, 2]]
        f_norms = np.cross(f_v1 - f_v0, f_v2 - f_v0)
        v_norms = (N_final[F_final[:,0]] + N_final[F_final[:,1]] + N_final[F_final[:,2]]) / 3.0
        dots = np.sum(f_norms * v_norms, axis=1)
        F_final[dots < 0] = F_final[dots < 0][:, [0, 2, 1]]
        
        # 6. Final topological cleanup (remove unreferenced vertices)
        # This is where the 'Smooth Weld' actually consolidates the index buffer.
        unique_v, remap_indices = np.unique(F_final, return_inverse=True)
        V_final = V_final[unique_v]
        # Store high-precision reference normals for orientation alignment
        N_ref = N_final[unique_v] if N_final is not None else None
        F_final = remap_indices.reshape(F_final.shape).astype(np.int32)
        
        # 7. Generate Final Vertex-Based FID for output
        # Map each vertex to the CAD face of its first incident triangle.
        FID_final = np.zeros(V_final.shape[0], dtype=np.int32)
        v_to_f_map = {int(v): i for i, f in enumerate(F_final) for v in f}
        for vid in range(V_final.shape[0]):
            if vid in v_to_f_map:
                FID_final[vid] = tri_face_work[v_to_f_map[vid]]

        # Safety Residual sync (if present)
        RES_final = np.zeros(V_final.shape[0]) 

        # 8. Recompute vertex normals from post-weld topology and optionally relax
        # (edge-preserving: blocked across sharp edges).
        # Since the topology is now welded, incident triangles from BOTH patches 
        # contribute to the normal at the seam, ensuring perfect G1 continuity.
        N_geo = _compute_vertex_normals_from_faces(V_final, F_final)
        
        if N_ref is not None and N_ref.shape == N_geo.shape:
            # Keep orientation coherent with analytical CAD normals and blend slightly.
            dots = np.sum(N_geo * N_ref, axis=1)
            flip_mask = dots < 0.0
            if np.any(flip_mask):
                N_geo[flip_mask] *= -1.0
            # Blend analytical precision with topological smoothness
            N_final = 0.85 * N_geo + 0.15 * N_ref
        else:
            N_final = N_geo

        sharp_edges_final = _find_sharp_edges(
            V_final,
            F_final,
            feature_lines=feature_lines,
            angle_threshold_deg=float(max(0.0, shade_sharp_deg)),
        )
        if int(normal_relax_iters) > 0 and float(normal_relax_strength) > 0.0:
            N_final = _relax_vertex_normals_edge_preserving(
                N_final,
                F_final,
                sharp_edges=sharp_edges_final,
                iters=int(normal_relax_iters),
                strength=float(normal_relax_strength),
                ignore_sharp=bool(normal_relax_global),
            )

        # 9. Normal Re-Normalization
        mags = np.linalg.norm(N_final, axis=1, keepdims=True)
        N_final = np.divide(N_final, mags, out=N_final, where=mags > 1e-12)
        
        print(f"[Denoise] Stitched {n_verts_before_weld - V_final.shape[0]} vertices across patches.")
        print(f"[Denoise] Pruned {sliver_count} sliver/needle triangles.")
        
        if flip_normals:
            N_final *= -1.0

    return Phase5Data(
        V_quad=V_final,
        Q=F_final,
        projected_normals=N_final,
        projected_face_indices=FID_final,
        projection_residual_to_face=RES_final,
        shading_diagnostics={
            "harmonization_count": int(harmonization_count),
            "antipodal_flip_count": int(antipodal_flip_count),
            "seam_sign_sync_count": int(seam_sign_sync_count),
            "smooth_weld_merge_count": int(smooth_weld_merge_count),
            "chamfer_edge_collapse_count": int(chamfer_edge_collapse_count),
            "normal_relax_iters": int(max(0, normal_relax_iters)),
            "normal_relax_strength": float(max(0.0, normal_relax_strength)),
            "max_tangent_normal_jump_deg": float(max_jump),
            "max_all_boundary_normal_jump_deg": float(max_jump_all),
            "max_sharp_boundary_normal_jump_deg": float(max_jump_sharp),
            "edge_snapping_count": int(snapping_count),
            "degenerate_faces_removed": int(degenerate_count),
        }
    )


def _remove_sliver_faces(
    V: np.ndarray, F: np.ndarray, max_aspect: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove triangles with excessively high aspect ratios (needles)."""
    if F.size == 0:
        return F, np.zeros(0, dtype=bool)
        
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    
    # Perimeter
    s = (e0 + e1 + e2) / 2.0
    # Area (Heron)
    area = np.sqrt(np.maximum(0, s * (s - e0) * (s - e1) * (s - e2)))
    
    # Aspect Ratio heuristic: (longest edge)^2 / (4 * area * sqrt(3))
    # Or simpler: max_edge / min_altitude.
    # We use: max_edge^2 / area
    max_e = np.maximum(e0, np.maximum(e1, e2))
    # Area might be 0 for degenerates, dealt with elsewhere.
    # sliver_metric = max_e^2 / area
    # For a perfect triangle, max_e^2 / area is approx 2.3
    # For a sliver with max_aspect=1000, max_e^2 / area is approx 1000
    
    aspect = np.divide(max_e**2, area, out=np.zeros_like(max_e), where=area > 1e-15)
    
    mask = aspect < max_aspect
    return F[mask], mask


def _transfer_face_ids(
    old_V: np.ndarray,
    old_F: np.ndarray,
    old_face_ids: np.ndarray,
    new_V: np.ndarray,
    new_F: np.ndarray,
) -> np.ndarray:
    """Transfer face provenance IDs from old mesh to new mesh via spatial proximity.
    
    Used after remeshing to maintain the link to original CAD faces.
    """
    from scipy.spatial import cKDTree
    
    if old_F.size == 0 or new_F.size == 0:
        return np.zeros(new_F.shape[0], dtype=np.int32)
        
    # Compute centroids of old faces
    old_centroids = np.mean(old_V[old_F], axis=1)
    tree = cKDTree(old_centroids)
    
    # Compute centroids of new faces
    new_centroids = np.mean(new_V[new_F], axis=1)
    
    # Query nearest old face for each new face
    _, indices = tree.query(new_centroids, k=1)
    
    # Transfer the IDs
    return old_face_ids[indices].copy()
