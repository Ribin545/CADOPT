from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import polyscope as ps
from polyscope import imgui as psim

from .viewer import init_viewer


def _parse_obj(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Parse an OBJ with 'v', optional 'vn', and 'f' records into triangle soup."""
    verts: List[List[float]] = []
    vnormals: List[List[float]] = []
    tris: List[List[int]] = []
    tri_normals: List[List[float]] = []

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        if s.startswith("v "):
            parts = s.split()
            if len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            continue

        if s.startswith("vn "):
            parts = s.split()
            if len(parts) >= 4:
                vnormals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            continue

        if s.startswith("f "):
            parts = s.split()[1:]
            if len(parts) < 3:
                continue

            face_vids: List[int] = []
            face_nids: List[int | None] = []
            for tok in parts:
                # Supports forms: v, v/vt, v//vn, v/vt/vn
                chunks = tok.split("/")
                vtok = chunks[0].strip() if len(chunks) > 0 else ""
                if not vtok:
                    continue
                vi = int(vtok)
                if vi < 0:
                    # Negative indices are relative to end.
                    vi = len(verts) + vi + 1
                face_vids.append(vi - 1)  # OBJ is 1-based.

                nidx = None
                if len(chunks) >= 3 and chunks[2].strip():
                    ni = int(chunks[2].strip())
                    if ni < 0:
                        ni = len(vnormals) + ni + 1
                    nidx = ni - 1
                face_nids.append(nidx)

            if len(face_vids) < 3:
                continue

            # Fan triangulate polygons (tri/quad/ngon).
            a = face_vids[0]
            for i in range(1, len(face_vids) - 1):
                tris.append([a, face_vids[i], face_vids[i + 1]])
                if len(face_nids) == len(face_vids):
                    n0 = face_nids[0]
                    n1 = face_nids[i]
                    n2 = face_nids[i + 1]
                    if n0 is not None and n1 is not None and n2 is not None:
                        tri_normals.append([n0, n1, n2])

    if not verts or not tris:
        raise RuntimeError(f"OBJ has no valid mesh data: {path}")

    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(tris, dtype=np.int32)
    VN = np.asarray(vnormals, dtype=np.float64) if vnormals else None
    return V, F, VN


class OutputMeshViewerApp:
    """Minimal viewer for final projected output mesh."""

    def __init__(self, project_root: Path, mesh_path: Path | None = None) -> None:
        self.project_root = Path(project_root)
        self.default_mesh_path = self.project_root / "Demo" / "output" / "latest_phase5.obj"
        self.mesh_path = Path(mesh_path) if mesh_path is not None else self.default_mesh_path
        self.status = ""
        self._mesh_handle = None

    def _load_and_show(self) -> None:
        if not self.mesh_path.exists():
            raise FileNotFoundError(f"Output mesh not found: {self.mesh_path}")

        V, F, VN = _parse_obj(self.mesh_path)
        if self._mesh_handle is not None:
            self._mesh_handle.remove()

        m = ps.register_surface_mesh("Final Output Mesh", V, F)
        m.set_smooth_shade(True)
        if VN is not None and VN.shape[0] == V.shape[0]:
            m.add_vector_quantity("OBJ Vertex Normals", VN, enabled=False)
        self._mesh_handle = m
        self.status = f"Loaded {self.mesh_path} (|V|={V.shape[0]}, |F|={F.shape[0]})"

    @staticmethod
    def _safe_begin(title: str) -> bool:
        """Handle imgui Begin API variants across polyscope versions."""
        out = psim.Begin(title, True)
        if isinstance(out, tuple):
            expanded, _is_open = out
            return bool(expanded)
        return bool(out)

    def _ui_callback(self) -> None:
        if not self._safe_begin("CADOPT Output Viewer"):
            psim.End()
            return

        psim.TextWrapped("Minimal viewer mode: displays only the final output mesh OBJ.")
        psim.Separator()
        psim.TextWrapped(f"Mesh path: {self.mesh_path}")

        if psim.Button("Reload latest output mesh"):
            self.mesh_path = self.default_mesh_path
            try:
                self._load_and_show()
            except Exception as exc:
                self.status = f"ERROR: {exc}"

        if self.status:
            psim.Separator()
            psim.TextWrapped(self.status)

        psim.End()

    def run(self) -> None:
        init_viewer()
        set_default_panels = getattr(ps, "set_build_default_gui_panels", None)
        if callable(set_default_panels):
            try:
                set_default_panels(False)
            except Exception:
                pass

        try:
            self._load_and_show()
        except Exception as exc:
            self.status = f"ERROR: {exc}"

        ps.set_user_callback(self._ui_callback)
        ps.show()
