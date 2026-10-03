from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCC.Core.GCPnts import GCPnts_QuasiUniformAbscissa
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON
from OCC.Core.TopExp import TopExp_Explorer, topexp_MapShapesAndAncestors
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_ListIteratorOfListOfShape,
)
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape, topods
from OCC.Core.gp import gp_Pnt, gp_Pnt2d

from .tessellation import compute_face_centroids
from .types import FeaturePolyline, Phase1Data, Phase2Data


def _finite_param_bounds(adaptor: BRepAdaptor_Surface) -> Tuple[float, float, float, float]:
    u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()

    if abs(float(u_max - u_min)) > 1e10:
        u_min, u_max = 0.0, float(2.0 * np.pi)
    if abs(float(v_max - v_min)) > 1e10:
        v_min, v_max = 0.0, float(2.0 * np.pi)
    return float(u_min), float(u_max), float(v_min), float(v_max)


def _extract_face_isoparm_polylines(
    face: TopoDS_Face,
    n_u: int = 3,
    n_v: int = 3,
    n_samples_per_iso: int = 64,
    include_planes: bool = False,
) -> List[np.ndarray]:
    """Extract trimmed U/V isoparms as polyline segments in 3D."""
    adaptor = BRepAdaptor_Surface(face)
    if (not include_planes) and adaptor.GetType() == GeomAbs_Plane:
        return []

    u_min, u_max, v_min, v_max = _finite_param_bounds(adaptor)
    classifier = BRepTopAdaptor_FClass2d(face, 1e-6)

    isoparms: List[np.ndarray] = []

    def _append_runs(uv_pairs: np.ndarray) -> None:
        current: List[List[float]] = []
        for u, v in uv_pairs:
            state = classifier.Perform(gp_Pnt2d(float(u), float(v)))
            if state in (TopAbs_IN, TopAbs_ON):
                p = adaptor.Value(float(u), float(v))
                current.append([p.X(), p.Y(), p.Z()])
            else:
                if len(current) >= 2:
                    isoparms.append(np.asarray(current, dtype=np.float64))
                current = []

        if len(current) >= 2:
            isoparms.append(np.asarray(current, dtype=np.float64))

    v_samples = np.linspace(v_min, v_max, max(8, int(n_samples_per_iso)))
    for u_fixed in np.linspace(u_min, u_max, n_u + 2)[1:-1]:
        uv_pairs = np.column_stack(
            [np.full_like(v_samples, fill_value=float(u_fixed), dtype=np.float64), v_samples]
        )
        _append_runs(uv_pairs)

    u_samples = np.linspace(u_min, u_max, max(8, int(n_samples_per_iso)))
    for v_fixed in np.linspace(v_min, v_max, n_v + 2)[1:-1]:
        uv_pairs = np.column_stack(
            [u_samples, np.full_like(u_samples, fill_value=float(v_fixed), dtype=np.float64)]
        )
        _append_runs(uv_pairs)

    return isoparms


def extract_face_isoparms(
    face: TopoDS_Face,
    n_u: int = 3,
    n_v: int = 3,
    n_samples_per_iso: int = 64,
    include_planes: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tangent-weighted directional samples extracted from trimmed face isoparms.

    Returns
    -------
    pts: (N,3)
        Segment midpoint positions in 3D.
    dirs: (N,3)
        Unit segment tangents.
    weights: (N,)
        Relative reliability/importance weights derived from local segment length.
    """
    polylines = _extract_face_isoparm_polylines(
        face,
        n_u=n_u,
        n_v=n_v,
        n_samples_per_iso=n_samples_per_iso,
        include_planes=include_planes,
    )
    if not polylines:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,))

    mids: List[np.ndarray] = []
    dirs: List[np.ndarray] = []
    w: List[float] = []

    for pts in polylines:
        if pts.shape[0] < 2:
            continue
        seg = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        valid = seg_len > 1e-12
        if not np.any(valid):
            continue

        seg = seg[valid]
        seg_len = seg_len[valid]
        p0 = pts[:-1][valid]
        p1 = pts[1:][valid]

        mids.append(0.5 * (p0 + p1))
        dirs.append(seg / seg_len[:, None])
        w.append(seg_len)

    if not mids:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,))

    pts_arr = np.vstack(mids).astype(np.float64)
    dirs_arr = np.vstack(dirs).astype(np.float64)
    w_arr = np.concatenate(w).astype(np.float64)

    med = float(np.median(w_arr[w_arr > 1e-12])) if np.any(w_arr > 1e-12) else 1.0
    if med <= 0.0:
        med = 1.0
    w_arr = np.clip(w_arr / med, 0.25, 4.0)
    return pts_arr, dirs_arr, w_arr


def _downsample_directional_constraints(
    pts: np.ndarray,
    dirs: np.ndarray,
    weights: np.ndarray,
    voxel_size: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pts.shape[0] == 0:
        return pts, dirs, weights

    if voxel_size is None:
        bb_min = pts.min(axis=0)
        bb_max = pts.max(axis=0)
        diag = float(np.linalg.norm(bb_max - bb_min))
        voxel_size = max(1e-6, diag * 1e-3)

    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    key_arr = np.floor(pts / float(voxel_size)).astype(np.int64)
    for i, k in enumerate(key_arr):
        key = (int(k[0]), int(k[1]), int(k[2]))
        buckets.setdefault(key, []).append(i)

    out_pts: List[np.ndarray] = []
    out_dirs: List[np.ndarray] = []
    out_w: List[float] = []

    for indices in buckets.values():
        idx = np.asarray(indices, dtype=np.int32)
        w = np.asarray(weights[idx], dtype=np.float64)
        if not np.any(w > 0):
            w = np.ones_like(w)

        p = np.average(pts[idx], axis=0, weights=w)

        d_ref = dirs[idx[0]].copy()
        d_ref /= max(np.linalg.norm(d_ref), 1e-12)
        d_acc = np.zeros((3,), dtype=np.float64)
        for ii, wi in zip(idx, w):
            d = dirs[ii].copy()
            nd = np.linalg.norm(d)
            if nd <= 1e-12:
                continue
            d /= nd
            if np.dot(d, d_ref) < 0.0:
                d = -d
            d_acc += wi * d

        nd_acc = np.linalg.norm(d_acc)
        if nd_acc <= 1e-12:
            continue

        out_pts.append(p)
        out_dirs.append(d_acc / nd_acc)
        out_w.append(float(np.sum(w)))

    if not out_pts:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,))
    return np.asarray(out_pts), np.asarray(out_dirs), np.asarray(out_w)


def extract_face_flow_constraints(face: TopoDS_Face, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy compatibility API returning raw point-direction samples per face."""
    from OCC.Core.gp import gp_Vec

    adaptor = BRepAdaptor_Surface(face)
    u_min, u_max, v_min, v_max = _finite_param_bounds(adaptor)
    classifier = BRepTopAdaptor_FClass2d(face, 1e-6)

    pts: List[List[float]] = []
    dirs: List[np.ndarray] = []
    samples_per_dim = max(2, int(np.sqrt(max(4, n_samples))))

    for u in np.linspace(u_min, u_max, samples_per_dim):
        for v in np.linspace(v_min, v_max, samples_per_dim):
            if classifier.Perform(gp_Pnt2d(float(u), float(v))) not in (TopAbs_IN, TopAbs_ON):
                continue

            p = gp_Pnt()
            d1u = gp_Vec()
            d1v = gp_Vec()
            adaptor.D1(float(u), float(v), p, d1u, d1v)

            du = np.array([d1u.X(), d1u.Y(), d1u.Z()], dtype=np.float64)
            nu = np.linalg.norm(du)
            if nu <= 1e-12:
                dv = np.array([d1v.X(), d1v.Y(), d1v.Z()], dtype=np.float64)
                nv = np.linalg.norm(dv)
                if nv <= 1e-12:
                    continue
                du = dv / nv
            else:
                du = du / nu

            pts.append([p.X(), p.Y(), p.Z()])
            dirs.append(du)

    if not pts:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.asarray(pts, dtype=np.float64), np.asarray(dirs, dtype=np.float64)


def extract_assembly_flow_constraints(
    shape: TopoDS_Shape,
    n_u: int = 3,
    n_v: int = 3,
    n_samples_per_iso: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect tangent-weighted analytical constraints from all faces in an assembly."""
    all_pts: List[np.ndarray] = []
    all_dirs: List[np.ndarray] = []
    all_w: List[np.ndarray] = []

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        pts, dirs, weights = extract_face_isoparms(
            face,
            n_u=n_u,
            n_v=n_v,
            n_samples_per_iso=n_samples_per_iso,
            include_planes=False,
        )
        if pts.shape[0] == 0:
            pts, dirs = extract_face_flow_constraints(face, n_samples=64)
            if pts.shape[0] > 0:
                weights = np.ones((pts.shape[0],), dtype=np.float64)

        if pts.shape[0] > 0:
            all_pts.append(pts)
            all_dirs.append(dirs)
            all_w.append(weights)

        exp.Next()

    if not all_pts:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,))

    pts = np.vstack(all_pts)
    dirs = np.vstack(all_dirs)
    w = np.concatenate(all_w)
    return _downsample_directional_constraints(pts, dirs, w)


def _safe_normal_at_point(face: TopoDS_Face, p_xyz: np.ndarray) -> Optional[np.ndarray]:
    """Project 3D point to face UV and evaluate OCC exact normal."""
    surf = BRep_Tool.Surface(face)
    sas = ShapeAnalysis_Surface(surf)
    uv = sas.ValueOfUV(gp_Pnt(float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2])), 1e-7)
    props = GeomLProp_SLProps(surf, uv.X(), uv.Y(), 1, 1e-7)
    if props.IsNormalDefined():
        n = props.Normal()
        arr = np.array([n.X(), n.Y(), n.Z()], dtype=np.float64)
        nn = np.linalg.norm(arr)
        if nn > 1e-12:
            return arr / nn
    return None


def _edge_sample_polyline(edge, n_samples_target: int = 64) -> np.ndarray:
    """Discretize a B-Rep edge into a polyline in 3D."""
    curve = BRepAdaptor_Curve(edge)
    first = curve.FirstParameter()
    last = curve.LastParameter()

    if not np.isfinite(first) or not np.isfinite(last):
        first, last = 0.0, 1.0

    pts: List[np.ndarray] = []
    try:
        abscissa = GCPnts_QuasiUniformAbscissa(curve, max(2, n_samples_target))
        if abscissa.IsDone() and abscissa.NbPoints() >= 2:
            for i in range(1, abscissa.NbPoints() + 1):
                u = abscissa.Parameter(i)
                p = curve.Value(u)
                pts.append(np.array([p.X(), p.Y(), p.Z()], dtype=np.float64))
    except Exception:
        pts = []

    if len(pts) < 2:
        us = np.linspace(first, last, max(2, n_samples_target))
        pts = []
        for u in us:
            try:
                p = curve.Value(float(u))
                pts.append(np.array([p.X(), p.Y(), p.Z()], dtype=np.float64))
            except Exception:
                continue

    if len(pts) < 2:
        try:
            p0 = curve.Value(float(first))
            p1 = curve.Value(float(last))
            pts = [
                np.array([p0.X(), p0.Y(), p0.Z()], dtype=np.float64),
                np.array([p1.X(), p1.Y(), p1.Z()], dtype=np.float64),
            ]
        except Exception:
            pts = [np.zeros((3,), dtype=np.float64), np.array([1e-9, 0.0, 0.0], dtype=np.float64)]

    return np.asarray(pts, dtype=np.float64)


def extract_feature_lines(
    phase1: Phase1Data,
    dihedral_threshold_deg: float = 40.0,
) -> Phase2Data:
    """Phase 2: extract hard-edge and analytical flow guidance lines + tangential constraints."""
    shape = phase1.shape

    edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp_MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)

    polylines: List[FeaturePolyline] = []
    seen_edges = set()
    constrained_face_indices: List[int] = []
    constrained_vectors: List[np.ndarray] = []

    face_centroids = compute_face_centroids(phase1.V, phase1.F)
    centroid_tree = cKDTree(face_centroids)

    edge_exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_exp.More():
        edge = topods.Edge(edge_exp.Current())

        ancestors = edge_to_faces.FindFromKey(edge)
        adj_faces = []
        it = TopTools_ListIteratorOfListOfShape(ancestors)
        while it.More():
            adj_faces.append(topods.Face(it.Value()))
            it.Next()

        is_boundary = len(adj_faces) == 1
        is_sharp = False

        edge_poly = _edge_sample_polyline(edge, n_samples_target=16)
        if edge_poly.shape[0] >= 2 and len(adj_faces) >= 2:
            mid = edge_poly[edge_poly.shape[0] // 2]
            n0 = _safe_normal_at_point(adj_faces[0], mid)
            n1 = _safe_normal_at_point(adj_faces[1], mid)
            if n0 is not None and n1 is not None:
                c = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(c)))
                is_sharp = angle >= float(dihedral_threshold_deg)

        if is_boundary or is_sharp:
            if edge_poly.shape[0] < 2:
                edge_exp.Next()
                continue

            p_start = tuple(np.round(edge_poly[0], 3))
            p_end = tuple(np.round(edge_poly[-1], 3))
            edge_key = (frozenset([p_start, p_end]), round(np.linalg.norm(edge_poly[-1] - edge_poly[0]), 3))
            if edge_key in seen_edges:
                edge_exp.Next()
                continue
            seen_edges.add(edge_key)

            edge_len = float(np.sum(np.linalg.norm(np.diff(edge_poly, axis=0), axis=1)))
            if edge_len < 1e-3:
                edge_exp.Next()
                continue

            kind = "boundary" if is_boundary else "sharp"
            polylines.append(FeaturePolyline(points=edge_poly, edge_kind=kind))

            mids: List[np.ndarray] = []
            tans: List[np.ndarray] = []
            for i in range(edge_poly.shape[0] - 1):
                p0 = edge_poly[i]
                p1 = edge_poly[i + 1]
                t = p1 - p0
                nt = np.linalg.norm(t)
                if nt < 1e-12:
                    continue
                mids.append(0.5 * (p0 + p1))
                tans.append(t / nt)

            if mids:
                mids_arr = np.asarray(mids, dtype=np.float64)
                _, nn_idx = centroid_tree.query(mids_arr, k=1)
                for fid, t in zip(nn_idx.tolist(), tans):
                    constrained_face_indices.append(int(fid))
                    constrained_vectors.append(t)

        edge_exp.Next()

    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    sampled_count = 0
    while face_exp.More():
        face = topods.Face(face_exp.Current())
        face_isoparms = _extract_face_isoparm_polylines(face, n_u=3, n_v=3, n_samples_per_iso=64)
        if not face_isoparms:
            face_exp.Next()
            continue

        for pts in face_isoparms:
            if pts.shape[0] < 2:
                continue
            polylines.append(FeaturePolyline(points=pts, edge_kind="flow"))
            sampled_count += 1

            mids: List[np.ndarray] = []
            tans: List[np.ndarray] = []
            for i in range(pts.shape[0] - 1):
                p0, p1 = pts[i], pts[i + 1]
                t = p1 - p0
                nt = np.linalg.norm(t)
                if nt > 1e-12:
                    mids.append(0.5 * (p0 + p1))
                    tans.append(t / nt)

            if mids:
                mids_arr = np.asarray(mids, dtype=np.float64)
                _, nn_idx = centroid_tree.query(mids_arr, k=1)
                for fid, t in zip(nn_idx.tolist(), tans):
                    constrained_face_indices.append(int(fid))
                    constrained_vectors.append(t)

        face_exp.Next()

    cfi = np.asarray(constrained_face_indices, dtype=np.int32)
    cv = np.asarray(constrained_vectors, dtype=np.float64)

    print(
        f"[DEBUG] Phase 2: SUCCESS. Extracted {len(polylines)} polylines "
        f"including {sampled_count} NURBS isoparms."
    )
    return Phase2Data(polylines=polylines, constrained_face_indices=cfi, constrained_vectors=cv)
