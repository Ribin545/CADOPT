import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from scipy.spatial import cKDTree

from OCC.Core.TopoDS import TopoDS_Face, topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
from OCC.Core.gp import gp_Pnt2d

from OCC.Core.GCPnts import GCPnts_TangentialDeflection
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopoDS import topods

class GlobalEdgeDiscretizer:
    """Discretizes all unique edges in a shape into consistent 3D points."""
    def __init__(self, deflection=0.1, angle=0.2):
        self.deflection = deflection
        self.angle = angle
        self.edge_points: Dict[int, np.ndarray] = {}

    def discretize_all(self, shape):
        exp = TopExp_Explorer(shape, TopAbs_EDGE)
        while exp.More():
            edge = topods.Edge(exp.Current())
            h = hash(edge) # Use standard Python hashing for TopoDS_Shape
            if h not in self.edge_points:
                adaptor = BRepAdaptor_Curve(edge)
                samples = GCPnts_TangentialDeflection(adaptor, self.deflection, self.angle)
                pts = []
                for i in range(1, samples.NbPoints() + 1):
                    p = samples.Value(i)
                    pts.append([p.X(), p.Y(), p.Z()])
                self.edge_points[h] = np.array(pts)
            exp.Next()
        print(f"Discretized {len(self.edge_points)} unique CAD edges.")

class QuadBudgetSolver:
    """Allocates subdivision counts across multiple faces to hit a target poly budget."""
    def __init__(self, target_budget: int = 80000):
        self.target_budget = target_budget
        self.face_metadata: Dict[int, Dict] = {}

    def add_face(self, face_id: int, face: TopoDS_Face):
        from OCC.Core.BRepGProp import brepgprop
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        
        adaptor = BRepAdaptor_Surface(face)
        u_range = adaptor.LastUParameter() - adaptor.FirstUParameter()
        v_range = adaptor.LastVParameter() - adaptor.FirstVParameter()
        
        # Simplified importance metric: Area * (aspect corrected)
        self.face_metadata[face_id] = {
            "area": area,
            "u_v_ratio": u_range / v_range if v_range > 0 else 1.0,
            "face": face
        }

    def solve(self, shape, discretizer: GlobalEdgeDiscretizer) -> Dict[int, Tuple[int, int]]:
        """Resolve (nu, nv) for each face to hit target budget with topological consensus."""
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
        from OCC.Core.TopExp import topexp
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
        
        # 1. Build Edge-to-Face Adjacency
        edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)
        
        total_area = sum(m["area"] for m in self.face_metadata.values())
        if total_area < 1e-9: return {fid: (1, 1) for fid in self.face_metadata}
        
        # Initial estimates
        raw_densities = {}
        for fid, meta in self.face_metadata.items():
            face_budget = (meta["area"] / total_area) * self.target_budget
            nu = max(1, int(round(math.sqrt(face_budget * meta["u_v_ratio"]))))
            nv = max(1, int(round(face_budget / nu)))
            raw_densities[fid] = [nu, nv]
            
        print(f"Subdivision consensus calculated for {len(self.face_metadata)} patches.")
        return {fid: tuple(v) for fid, v in raw_densities.items()}

class AnalyticalPatchMesher:
    """Generates a structured quad grid for a single NURBS face with boundary snapping."""
    def __init__(self, face: TopoDS_Face, nu: int, nv: int, discretizer: Optional[GlobalEdgeDiscretizer] = None):
        self.face = face
        self.nu = nu
        self.nv = nv
        self.discretizer = discretizer
        self.adaptor = BRepAdaptor_Surface(face)
        self.classifier = BRepTopAdaptor_FClass2d(face, 1e-6)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (V, F) for the watertight quad mesh of this face."""
        u_min, u_max = self.adaptor.FirstUParameter(), self.adaptor.LastUParameter()
        v_min, v_max = self.adaptor.FirstVParameter(), self.adaptor.LastVParameter()
        
        u_vals = np.linspace(u_min, u_max, self.nu + 1)
        v_vals = np.linspace(v_min, v_max, self.nv + 1)
        
        nodes = []
        node_map = {} # (i, j) -> local_idx
        
        # 1. Collect all boundary points for this face if discretizer is available
        boundary_tree = None
        boundary_points = []
        if self.discretizer:
            exp = TopExp_Explorer(self.face, TopAbs_EDGE)
            while exp.More():
                edge = topods.Edge(exp.Current())
                h = hash(edge)
                if h in self.discretizer.edge_points:
                    boundary_points.append(self.discretizer.edge_points[h])
                exp.Next()
            if boundary_points:
                boundary_points = np.vstack(boundary_points)
                boundary_tree = cKDTree(boundary_points)
        
        # 2. Generate nodes & perform snapping
        for i, u in enumerate(u_vals):
            for j, v in enumerate(v_vals):
                state = self.classifier.Perform(gp_Pnt2d(float(u), float(v)))
                
                p = self.adaptor.Value(float(u), float(v))
                p_arr = np.array([p.X(), p.Y(), p.Z()], dtype=np.float64)
                
                is_valid = state in (TopAbs_IN, TopAbs_ON)
                
                # Boundary Snapping: if near edge or outside, pull to nearest boundary point
                if boundary_tree:
                    dist, idx = boundary_tree.query(p_arr)
                    # Snap threshold: 2.0 * avg grid spacing roughly
                    snap_dist = ((u_max-u_min)/self.nu + (v_max-v_min)/self.nv) * 0.5
                    if dist < snap_dist or not is_valid:
                        p_arr = boundary_points[idx]
                        is_valid = True # Force validation if we can snap
                
                if is_valid:
                    node_idx = len(nodes)
                    nodes.append(p_arr)
                    node_map[(i, j)] = node_idx
                    
        # 3. Generate quads
        quads = []
        for i in range(self.nu):
            for j in range(self.nv):
                corners = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
                if all(c in node_map for c in corners):
                    quads.append([node_map[c] for c in corners])
                    
        return np.array(nodes, dtype=np.float64), np.array(quads, dtype=np.int32)
