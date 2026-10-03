from __future__ import annotations

from typing import List, Set

import numpy as np

from .types import Phase1Data, Phase3Data, Phase4Data


def detect_singularities(phase1: Phase1Data, phase3: Phase3Data) -> Phase4Data:
    """Phase 4 core: locate singularity-like poles from discrete field curl.

    Prototype strategy:
    - Treat solved vertex angle field as a discrete potential.
    - Around each vertex, compare local variation to neighborhood average.
    - Large positive/negative residuals are tagged as valence-5/valence-3 proxies.
    """
    V = phase1.V
    F = phase1.F
    theta = phase3.vertex_angles

    n_vertices = V.shape[0]
    neighbors: List[Set[int]] = [set() for _ in range(n_vertices)]
    for tri in F:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])

    residual = np.zeros((n_vertices,), dtype=np.float64)
    for i in range(n_vertices):
        nb = list(neighbors[i])
        if not nb:
            continue
        nb_mean = np.mean(theta[nb])
        residual[i] = theta[i] - nb_mean

    scale = float(np.std(residual) + 1e-12)
    score = residual / scale

    # Heuristic thresholds for poles.
    is_neg = score < -2.5  # valence-3 style
    is_pos = score > 2.5   # valence-5 style

    idx = np.where(is_neg | is_pos)[0]
    if idx.size == 0:
        return Phase4Data(
            singular_points=np.zeros((0, 3), dtype=np.float64),
            singular_types=np.zeros((0,), dtype=np.int32),
        )

    pts = V[idx]
    types = np.where(is_neg[idx], -1, 1).astype(np.int32)

    return Phase4Data(singular_points=pts, singular_types=types)
