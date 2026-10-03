from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Phase1Data:
    """Dense proxy mesh plus provenance back to OCC faces."""

    shape: Any
    occ_faces: List[Any]
    V: np.ndarray  # (n, 3)
    F: np.ndarray  # (m, 3)
    tri_to_face_idx: np.ndarray  # (m,)
    VN: Optional[np.ndarray] = None  # (n, 3) Analytical vertex normals
    UV: Optional[np.ndarray] = None  # (n, 2) Analytical UV coordinates


@dataclass
class FeaturePolyline:
    """One extracted feature edge represented as a sampled polyline."""

    points: np.ndarray
    edge_kind: str  # 'boundary' or 'sharp'


@dataclass
class Phase2Data:
    polylines: List[FeaturePolyline]
    constrained_face_indices: np.ndarray
    constrained_vectors: np.ndarray


@dataclass
class Phase3Data:
    face_centroids: np.ndarray
    face_vectors: np.ndarray
    face_tangent_u: np.ndarray
    face_tangent_v: np.ndarray
    face_angles: np.ndarray
    vertex_angles: np.ndarray


@dataclass
class Phase4Data:
    singular_points: np.ndarray
    singular_types: np.ndarray  # -1 => valence 3, +1 => valence 5


@dataclass
class RemeshSettings:
    target_edge_length: float = 0.5
    max_iters: int = 5
    shade_sharp_deg: float = 45.0
    smooth_weld_enable: bool = True
    smooth_weld_tol: float = 1e-6
    smooth_weld_angle_deg: float = 35.0
    normal_relax_iters: int = 0
    normal_relax_strength: float = 0.2


@dataclass
class UVAtlasSettings:
    """Settings for single-canvas and multi-tile UDIM UV atlas generation."""

    mode: str = "SingleTile"  # Options: "SingleTile", "UDIM-Auto", "UDIM-Bucketed"
    max_udim_cols: int = 10  # Maximum UDIM tiles per row (standard 10)
    bucket_thresholds_mm2: List[float] = field(
        default_factory=lambda: [1000.0, 100.0, 10.0]
    )  # Cutoff surface area thresholds in mm^2 for UDIM-Bucketed mode
    target_tile_density: float = 0.80  # Target packing density ratio per tile


@dataclass
class Phase5Data:
    V_quad: np.ndarray
    Q: np.ndarray
    projected_normals: np.ndarray
    projected_face_indices: np.ndarray
    projection_residual_to_face: np.ndarray
    shading_diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppState:
    step_path: Optional[str] = None
    phase1: Optional[Phase1Data] = None
    phase2: Optional[Phase2Data] = None
    phase3: Optional[Phase3Data] = None
    phase4: Optional[Phase4Data] = None
    phase5: Optional[Phase5Data] = None
    messages: List[str] = field(default_factory=list)
    handles: Dict[str, Any] = field(default_factory=dict)

    def log(self, message: str) -> None:
        self.messages.append(message)
        print(message)
