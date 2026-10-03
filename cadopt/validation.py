from __future__ import annotations
import numpy as np
import trimesh
from typing import Dict, List, Any, Optional
from loguru import logger

class GeometryValidator:
    """Headless QA Auditor for high-fidelity CAD meshes.
    
    Validates mathematical integrity of vertex normals and provides 
    industrial-grade heuristic checks.
    """

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.report = {
            "pass": True,
            "errors": {},
            "metrics": {}
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """Execute the 4 industrial heuristic tests."""
        logger.info("[QA] Starting Geometry Validation suite...")
        
        self._test_normalization()
        self._test_self_shadowing()
        self._test_degenerate_vectors()
        self._test_coplanar_drift()
        
        # Final status
        self.report["pass"] = len(self.report["errors"]) == 0
        if self.report["pass"]:
            logger.info("[QA] ALL TESTS PASSED.")
        else:
            logger.warning(f"[QA] VALIDATION FAILED: {len(self.report['errors'])} markers found.")
            
        return self.report

    def _test_self_shadowing(self) -> None:
        """Heuristic 1: Detect normals pointing 'inside' the face."""
        # trimesh handles face_normals automatically
        f_normals = self.mesh.face_normals
        v_indices = self.mesh.faces  # (M, 3)
        v_normals = self.mesh.vertex_normals
        
        # Check normalization of each vertex normal against its face normal
        bad_faces = []
        for i, face in enumerate(v_indices):
            fn = f_normals[i]
            for v_idx in face:
                vn = v_normals[v_idx]
                if np.dot(vn, fn) < -1e-5: # Small epsilon for curved edges
                    bad_faces.append(i)
                    break
        
        if bad_faces:
            self.report["errors"]["self_shadowing"] = list(set(bad_faces))

    def _test_degenerate_vectors(self) -> None:
        """Heuristic 2: Catch NaNs or Zero-Length vectors."""
        vn = self.mesh.vertex_normals
        magnitudes = np.linalg.norm(vn, axis=1)
        
        zero_indices = np.where(magnitudes < 1e-9)[0]
        nan_indices = np.where(np.isnan(magnitudes))[0]
        
        bad_indices = np.unique(np.concatenate([zero_indices, nan_indices]))
        if len(bad_indices) > 0:
            self.report["errors"]["degenerate_vectors"] = bad_indices.tolist()

    def _test_normalization(self) -> None:
        """Heuristic 3: Audit scaling drift and auto-correct."""
        vn = self.mesh.vertex_normals
        magnitudes = np.linalg.norm(vn, axis=1)
        
        drift_mask = np.abs(magnitudes - 1.0) > 1e-5
        drift_indices = np.where(drift_mask)[0]
        
        if len(drift_indices) > 0:
            logger.warning(f"[QA] Detected {len(drift_indices)} non-unit normals. Auto-correcting...")
            # Fallback: Auto-normalize
            safe_mags = np.maximum(magnitudes, 1e-12)
            self.mesh.vertex_normals = vn / safe_mags[:, np.newaxis]
            self.report["errors"]["normalization_drift"] = drift_indices.tolist()
            self.report["metrics"]["normalization_fixed"] = True

    def _test_coplanar_drift(self) -> None:
        """Heuristic 5: Detect 'wavy' shading on flat mechanical walls."""
        # Use trimesh grouping to find coplanar face clusters
        # we check normals within a 1e-4 tolerance
        groups = trimesh.grouping.group_rows(self.mesh.face_normals, digits=4)
        
        wavy_groups = []
        v_normals = self.mesh.vertex_normals
        
        for g_idx, face_indices in enumerate(groups):
            if len(face_indices) < 3: continue # Ignore tiny clusters
            
            # Extract all vertex normals associated with this group
            cluster_v_indices = np.unique(self.mesh.faces[face_indices])
            cluster_v_normals = v_normals[cluster_v_indices]
            
            # Measure variance
            variance = np.var(cluster_v_normals, axis=0).sum()
            if variance > 1e-6:
                wavy_groups.append({
                    "group_id": g_idx,
                    "variance": float(variance),
                    "face_count": len(face_indices)
                })
        
        if wavy_groups:
            self.report["errors"]["coplanar_drift"] = wavy_groups
