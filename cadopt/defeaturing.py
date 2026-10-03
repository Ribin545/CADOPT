from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Torus, GeomAbs_Cone, GeomAbs_Sphere
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Defeaturing
from OCC.Core.TopTools import TopTools_ListOfShape

def find_tiny_faces(shape: TopoDS_Shape, area_threshold: float) -> List[TopoDS_Face]:
    """Identify faces with surface area below the given threshold."""
    tiny_faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        if area < area_threshold:
            tiny_faces.append(face)
        exp.Next()
    return tiny_faces

def find_micro_fillets(shape: TopoDS_Shape, radius_threshold: float) -> List[TopoDS_Face]:
    """Identify curved faces (fillets) with radius below the given threshold."""
    fillet_faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        adaptor = BRepAdaptor_Surface(face)
        s_type = adaptor.GetType()
        
        radius = float('inf')
        if s_type == GeomAbs_Cylinder:
            radius = adaptor.Cylinder().Radius()
        elif s_type == GeomAbs_Torus:
            radius = min(adaptor.Torus().MajorRadius(), adaptor.Torus().MinorRadius())
        elif s_type == GeomAbs_Cone:
            # For cones, we check the radius at the middle of the V range
            v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2.0
            radius = adaptor.Cone().RefRadius() + v_mid * np.tan(adaptor.Cone().SemiAngle())
        elif s_type == GeomAbs_Sphere:
            radius = adaptor.Sphere().Radius()
            
        if radius < radius_threshold:
            fillet_faces.append(face)
        exp.Next()
    return fillet_faces

def apply_native_defeaturing(shape: TopoDS_Shape, faces_to_remove: List[TopoDS_Face]) -> TopoDS_Shape:
    """Mathematically remove features and heal the NURBS body."""
    if not faces_to_remove:
        return shape
        
    defeaturing = BRepAlgoAPI_Defeaturing()
    defeaturing.SetShape(shape)
    
    face_list = TopTools_ListOfShape()
    for f in faces_to_remove:
        face_list.Append(f)
        
    defeaturing.AddFacesToRemove(face_list)
    defeaturing.Build()
    
    if not defeaturing.IsDone():
        # Fallback: return original shape if defeaturing fails
        return shape
        
    return defeaturing.Shape()
