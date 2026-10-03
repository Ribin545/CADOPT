from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Optional

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_Sewing
from OCC.Core.ShapeFix import ShapeFix_Shape

from .io_step import load_step_shape
from .defeaturing import find_tiny_faces, find_micro_fillets, apply_native_defeaturing
from .tessellation import tessellate_shape_with_face_mapping, TessellationSettings
from .types import AppState, UVAtlasSettings
from .validation import GeometryValidator
from .uv_mapping import apply_xatlas_uvs, apply_xatlas_udim_uvs
import trimesh

import math

NATIVE_PRESETS = {
    "Draft": {
        "linear_deflection": 0.5,
        "angular_deflection": 0.8, # ~45 deg
        "is_relative": False,
        "sample_normals": False
    },
    "Balanced": {
        "linear_deflection": 0.1,
        "angular_deflection": 0.5, # ~28 deg
        "is_relative": False,
        "sample_normals": True
    },
    "High": {
        "linear_deflection": 0.05,
        "angular_deflection": 0.2, # ~11 deg
        "is_relative": True,
        "sample_normals": True
    },
    "Ultra-CAD": {
        "linear_deflection": 0.005,
        "angular_deflection": 0.1, # ~5.7 deg
        "is_relative": True,
        "sample_normals": True
    },
    "Anisotropic": {
        "linear_deflection": 0.5, # Loose linear to allow stretching
        "angular_deflection": 20.0 * math.pi / 180.0, # Tight angular for curvature
        "is_relative": True,
        "sample_normals": True
    }
}

def run_native_optimization_pipeline(
    step_path: Path,
    area_threshold: float = 0.0,
    radius_threshold: float = 0.0,
    linear_deflection: Optional[float] = None,
    angular_deflection: Optional[float] = None,
    preset_name: Optional[str] = None,
    validate: bool = False,
    unwrap: bool = False,
    uv_settings: Optional[UVAtlasSettings] = None
) -> Dict:
    """A streamlined NURBS-native optimization pipeline."""
    state = AppState(step_path=str(step_path))
    
    # Resolve settings from preset or defaults
    preset = NATIVE_PRESETS.get(preset_name, NATIVE_PRESETS["Balanced"])
    
    ld = linear_deflection if linear_deflection is not None else preset["linear_deflection"]
    ad = angular_deflection if angular_deflection is not None else preset["angular_deflection"]

    metrics = {
        "step_path": str(step_path),
        "phases": {},
        "total_time_s": 0.0,
        "preset": preset_name or "Custom"
    }
    
    t0_total = time.perf_counter()
    raw_shape = load_step_shape(step_path)
    current_shape = raw_shape
    
    # Phase 1: Defeaturing
    print(f"[LOG] Phase 1/5: Starting Native Defeaturing (area={area_threshold}, radius={radius_threshold})...")
    t0 = time.perf_counter()
    faces_to_remove = []
    if area_threshold > 0:
        faces_to_remove.extend(find_tiny_faces(current_shape, area_threshold))
    if radius_threshold > 0:
        faces_to_remove.extend(find_micro_fillets(current_shape, radius_threshold))
        
    if faces_to_remove:
        unique_faces = []
        seen = set()
        for f in faces_to_remove:
            if f not in seen:
                unique_faces.append(f)
                seen.add(f)
        
        current_shape = apply_native_defeaturing(current_shape, unique_faces)
        dt = time.perf_counter() - t0
        metrics["phases"]["defeaturing"] = {
            "ok": True,
            "time_s": dt,
            "removed_faces": len(unique_faces)
        }
    else:
        metrics["phases"]["defeaturing"] = {"ok": True, "time_s": 0.0, "removed_faces": 0}

    # Phase 2: Precision Sewing & Healing
    print(f"[LOG] Phase 2/5: Starting Precise BRep Sewing and Healing...")
    t0 = time.perf_counter()
    sewer = BRepOffsetAPI_Sewing(0.01)
    sewer.Add(current_shape)
    sewer.Perform()
    sewn_shape = sewer.SewedShape()
    
    fixer = ShapeFix_Shape(sewn_shape)
    fixer.Perform()
    healed_shape = fixer.Shape()
    dt = time.perf_counter() - t0
    metrics["phases"]["healing"] = {"ok": True, "time_s": dt}

    # Phase 3: Optimized Tessellation
    print(f"[LOG] Phase 3/5: Starting Optimized Tessellation (preset={preset_name})...")
    t0 = time.perf_counter()
    p1 = tessellate_shape_with_face_mapping(
        healed_shape,
        settings=TessellationSettings(
            linear_deflection=ld,
            angular_deflection=ad,
            is_relative=preset["is_relative"],
            sample_analytical_normals=preset["sample_normals"]
        )
    )
    dt = time.perf_counter() - t0
    state.phase1 = p1
    metrics["phases"]["tessellation"] = {
        "ok": True,
        "time_s": dt,
        "n_verts": int(p1.V.shape[0]),
        "n_tris": int(p1.F.shape[0])
    }

    # Phase 4: UV Unwrapping (Optional)
    if unwrap or uv_settings is not None:
        effective_uv_settings = uv_settings or UVAtlasSettings(mode="SingleTile")
        # PHASE 2: Feature Intelligence (Extract topological boundaries)
        print(f"[LOG] Phase 2/5: Extracting Feature Intelligence...")
        from .features import extract_feature_lines
        from .shading import precision_attribute_weld
        
        # Extract boundaries for validation and potential seam guidance
        phase2 = extract_feature_lines(p1)
        
        # PHASE 3: Precision Sealing
        # Weld within each patch but preserve boundaries (implicit by face grouping later)
        print(f"[LOG] Phase 3/5: Precision Attribute Welding...")
        v_welded, vn_welded, f_welded = precision_attribute_weld(p1.V, p1.VN, p1.F)

        # PHASE 4: Semantic UV Unwrapping & UDIM Multi-Tile Packing
        print(f"[LOG] Phase 4/5: Starting Semantic UV Unwrapping (xatlas UDIM mode={effective_uv_settings.mode})...")
        v_new, vn_new, f_new, uv_new = apply_xatlas_udim_uvs(
            v_welded, 
            vn_welded, 
            f_welded, 
            face_indices=p1.tri_to_face_idx,
            uv_settings=effective_uv_settings
        )
        p1.V = v_new
        p1.VN = vn_new
        p1.F = f_new
        p1.UV = uv_new
        dt = time.perf_counter() - t0
        metrics["phases"]["uv_unwrapping"] = {
            "ok": True,
            "time_s": dt,
            "n_verts_after_split": int(p1.V.shape[0]),
            "uv_mode": effective_uv_settings.mode
        }

    # Phase 5: Optional QA Validation
    if validate:
        print(f"[LOG] Phase 5/5: Starting Geometry Validation...")
        t0 = time.perf_counter()
        # Create trimesh with process=False to ignore welding/auto-normal
        tm = trimesh.Trimesh(
            vertices=p1.V, 
            faces=p1.F, 
            vertex_normals=p1.VN,
            process=False
        )
        validator = GeometryValidator(tm)
        report = validator.run_all_tests()
        dt = time.perf_counter() - t0
        metrics["phases"]["qa_validation"] = {
            "ok": report["pass"],
            "time_s": dt,
            "report": report
        }
    else:
        metrics["phases"]["qa_validation"] = {"ok": True, "time_s": 0.0}
    
    metrics["total_time_s"] = time.perf_counter() - t0_total
    return {"state": state, "metrics": metrics}
