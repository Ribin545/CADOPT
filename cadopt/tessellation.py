from __future__ import annotations

import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.IMeshTools import IMeshTools_Parameters
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepTools import breptools
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_Sewing
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods, TopoDS_Face
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Dir, gp_Pnt

from .types import Phase1Data

@dataclass
class TessellationSettings:
    linear_deflection: float = 0.5  # Absolute (mm)
    angular_deflection: float = 0.3490 # ~20 deg
    is_relative: bool = False       
    is_parallel: bool = False
    sample_analytical_normals: bool = False

def _pnt_to_np(pnt) -> np.ndarray:
    return np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float64)

def precision_attribute_weld(
    vertices: np.ndarray, 
    normals: np.ndarray, 
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Welds physical gaps and identical CAD normals, but respects sharp 90-degree 
    edges by keeping differing analytical normals split.
    """
    # 1. STRICT spatial rounding (0.1mm accuracy to snap CAD gaps)
    q_vertices = np.round(vertices, decimals=4) + 0.0
    
    # 2. HIGH-PRECISION normal rounding (Preserves smooth gradients but catches 1e-6 noise)
    q_normals = np.round(normals, decimals=4) + 0.0
    
    # 3. Stack and find unique combinations
    stacked_attributes = np.hstack((q_vertices, q_normals))
    unique_stacked, unique_indices, inverse_indices = np.unique(
        stacked_attributes, axis=0, return_index=True, return_inverse=True
    )
    
    # 4. Extract
    final_vertices = unique_stacked[:, 0:3]
    final_normals = unique_stacked[:, 3:6]
    
    # 5. Re-normalize to ensure the shader engine gets exactly 1.0 length vectors
    norms = np.linalg.norm(final_normals, axis=1)
    norms[norms == 0] = 1.0
    final_normals = final_normals / norms[:, np.newaxis]
    
    # 6. Re-index faces
    final_faces = inverse_indices[faces].reshape(-1, 3)
    
    return final_vertices, final_normals, final_faces

def tessellate_shape_with_face_mapping(
    shape: object, settings: Optional[TessellationSettings] = None
) -> Phase1Data:
    """Tessellate shape at the Pure Analytical Baseline.
    Target: High Speed (~10s) and Native Shading Integrity.
    """
    settings = settings or TessellationSettings()

    # 1. Geometry Cleanup
    breptools.Clean(shape)
    
    # 2. Precision Sewing (0.01mm)
    # Tighter sewing for the analytical baseline.
    sewer = BRepOffsetAPI_Sewing(0.01)
    sewer.Add(shape)
    sewer.Perform()
    sewn_shape = sewer.SewedShape()
    
    # 3. Shape Fix
    fixer = ShapeFix_Shape(sewn_shape)
    fixer.Perform()
    fixed_shape = fixer.Shape()
    
    # 4. Baseline Mesher Strategy
    ld = settings.linear_deflection
    ad = settings.angular_deflection
    is_rel = settings.is_relative
    par = settings.is_parallel
    print(f"[DEBUG] tessellation: ld={ld}, ad={ad}, is_rel={is_rel}")

    mesher = BRepMesh_IncrementalMesh(fixed_shape, ld, is_rel, ad, par)
    mesher.Perform()

    occ_faces: List[object] = []
    face_exp = TopExp_Explorer(fixed_shape, TopAbs_FACE)
    face_idx_tracker = 0
    
    raw_points = []
    raw_normals = []
    raw_tri_data = [] 
    
    while face_exp.More():
        face = topods.Face(face_exp.Current())
        occ_faces.append(face)

        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            has_uv = tri.HasUVNodes()
            
            # For analytical normals
            adaptor = None
            if settings.sample_analytical_normals:
                adaptor = BRepAdaptor_Surface(face)
            
            is_reversed = (face.Orientation() == TopAbs_REVERSED)
            
            for i in range(1, tri.NbTriangles() + 1):
                t = tri.Triangle(i)
                indices = list(t.Get()) # (i1, i2, i3)
                
                # Flip winding order if reversed to ensure consistent normals
                if is_reversed:
                    indices[1], indices[2] = indices[2], indices[1]
                
                for node_idx in indices:
                    # Position
                    pnt = tri.Node(node_idx).Transformed(trsf)
                    raw_points.append(_pnt_to_np(pnt))
                    
                    # Normal
                    if settings.sample_analytical_normals and has_uv:
                        uv = tri.UVNode(node_idx)
                        props = GeomLProp_SLProps(adaptor.Surface().Surface(), uv.X(), uv.Y(), 1, 1e-6)
                        if props.IsNormalDefined():
                            n = props.Normal()
                            n.Transform(trsf)
                            if is_reversed:
                                n.Reverse()
                            if trsf.IsNegative():
                                n.Reverse()
                            raw_normals.append([n.X(), n.Y(), n.Z()])
                        else:
                            raw_normals.append([0.0, 0.0, 1.0])
                    else:
                        raw_normals.append([0.0, 0.0, 1.0])
                
                raw_tri_data.append(face_idx_tracker)
        
        face_idx_tracker += 1
        face_exp.Next()

    if not raw_points:
        raise RuntimeError("No triangles produced from the shape.")

    V_raw = np.array(raw_points, dtype=np.float64)
    VN_raw = np.array(raw_normals, dtype=np.float64)
    F_soup = np.arange(V_raw.shape[0]).reshape(-1, 3)

    # 4. Pure Analytical Weld
    # This replaces all previous heuristics and decimation resampling.
    V_final, VN_final, F_final = precision_attribute_weld(V_raw, VN_raw, F_soup)
    
    # Map back tri_to_face (soup was 3 vertices per face)
    tri_to_face_idx = np.array(raw_tri_data, dtype=np.int32)

    return Phase1Data(
        shape=fixed_shape, 
        occ_faces=occ_faces, 
        V=V_final, 
        F=F_final, 
        tri_to_face_idx=tri_to_face_idx,
        VN=VN_final,
        UV=np.zeros((V_final.shape[0], 2)) # UVs intentionally zeroed in this baseline
    )


def compute_face_centroids(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    return (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0


def compute_face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    n = np.cross(e1, e2)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    nn = np.maximum(nn, 1e-12)
    return n / nn
